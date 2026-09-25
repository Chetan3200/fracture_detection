"""Pinned MedGemma inference and uncalibrated coordinate-token ranking scores.

Heavy dependencies are imported only when constructing MedGemmaPredictor.
No ground truth, confidence floor, box cap, or NMS is used here.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import re

MODEL_ID = "google/medgemma-1.5-4b-it"
VERSIONS = {"torch": "2.11.0+cu128", "transformers": "4.57.6", "peft": "0.18.1",
            "bitsandbytes": "0.49.2", "accelerate": "1.12.0"}
PROJECTIONS = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
SKIP_MODULES = ["vision_tower", "multi_modal_projector", "lm_head", "embed_tokens"]
WRAPPER = re.compile(r'\s*Final Answer:\s*```json\s*(.*?)\s*```\s*', re.S)


class FormatFailure(ValueError):
    """The model did not emit the required complete fenced JSON document."""


class SchemaFailure(ValueError):
    """The parsed answer violates the frozen output schema."""


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise SchemaFailure(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value):
    raise SchemaFailure(f"Nonfinite JSON constant: {value}")


def _space(text, position):
    while position < len(text) and text[position] in " \t\r\n":
        position += 1
    return position


def _coordinate_spans(payload, offset):
    """Visit ordered object members with JSONDecoder offsets, never string.find.

    Called only after full-document parsing and strict schema validation.
    JSONDecoder also handles escaped key spelling and reversed key order.
    """
    decoder = json.JSONDecoder()
    spans = []
    p = _space(payload, 0) + 1  # opening top-level [
    while True:
        p = _space(payload, p)
        if payload[p] == "]":
            break
        _require(payload[p] == "{", "Internal JSON span mapping disagreement.")
        p += 1
        while True:
            p = _space(payload, p)
            key, p = decoder.raw_decode(payload, p)
            p = _space(payload, p)
            _require(payload[p] == ":", "Missing colon in validated JSON.")
            start = _space(payload, p + 1)
            value, end = decoder.raw_decode(payload, start)
            if key == "box_2d":
                _require(isinstance(value, list), "Internal coordinate span disagreement.")
                spans.append((offset + start, offset + end))
            p = _space(payload, end)
            if payload[p] == "}":
                p = _space(payload, p + 1)
                break
            _require(payload[p] == ",", "Missing member separator in validated JSON.")
            p += 1
        if payload[p] == "]":
            break
        _require(payload[p] == ",", "Missing object separator in validated JSON.")
        p += 1
    return spans


def parse_answer(text):
    """Return (validated box objects, ordered coordinate-array character spans)."""
    match = WRAPPER.fullmatch(text)
    if match is None:
        raise FormatFailure("Missing/extra text around the required Final Answer JSON fence.")
    payload = match.group(1)
    try:
        boxes = json.loads(payload, object_pairs_hook=_unique_pairs, parse_constant=_reject_constant)
    except json.JSONDecodeError as exc:
        raise FormatFailure(f"Invalid JSON: {exc.msg}") from exc
    if not isinstance(boxes, list):
        raise SchemaFailure("Expected a JSON list.")
    for obj in boxes:
        if not isinstance(obj, dict) or set(obj) != {"label", "box_2d"}:
            raise SchemaFailure("Wrong object keys.")
        if obj["label"] != "fracture":
            raise SchemaFailure("Wrong label.")
        box = obj["box_2d"]
        if not isinstance(box, list) or len(box) != 4:
            raise SchemaFailure("Wrong box shape.")
        # Bounds first also reject huge Python integers without float conversion.
        if not all(type(x) in (int, float) and 0 <= x <= 1000 and math.isfinite(x) for x in box):
            raise SchemaFailure("Invalid coordinate.")
        if not (box[0] < box[2] and box[1] < box[3]):
            raise SchemaFailure("Degenerate/reversed box.")
    spans = _coordinate_spans(payload, match.start(1))
    _require(len(spans) == len(boxes), "Coordinate span count mismatch.")
    return boxes, spans


def decode_with_spans(tokenizer, ids, eos_ids):
    """Decode actual IDs; remove only a terminal EOS by identity.

    Token span i belongs to the same token as recorded logprob i. Prefix
    instability is a scoring error, never a model-format failure. Internal
    special tokens remain in the returned text for strict parsing to reject.
    """
    stop = set(eos_ids)
    end = len(ids) - 1 if ids and ids[-1] in stop else len(ids)
    previous = ""
    spans = []
    clean = ""
    for i in range(end):
        current = tokenizer.decode(ids[:i + 1], skip_special_tokens=False,
                                   clean_up_tokenization_spaces=False)
        _require(current.startswith(previous), f"Non-prefix-stable token decoding at generated token {i}.")
        spans.append((len(previous), len(current)))
        previous = current
        if i + 1 == end:
            clean = current
    # Explicit terminal deletion must agree with independently decoding kept IDs.
    expected = tokenizer.decode(ids[:end], skip_special_tokens=False,
                                clean_up_tokenization_spaces=False)
    _require(clean == expected, "Terminal EOS text mapping disagreement.")
    return clean, spans[:end]


def score_boxes(boxes, coordinate_spans, token_spans, logprobs, width, height):
    """Use each whole token with positive overlap with a coordinate-array span."""
    _require(len(boxes) == len(coordinate_spans), "Box/span mismatch.")
    _require(len(token_spans) <= len(logprobs), "Missing recorded token scores.")
    _require(all(math.isfinite(x) and x <= 0 for x in logprobs), "Invalid token logprob.")
    xyxy, scores = [], []
    for obj, (left, right) in zip(boxes, coordinate_spans):
        selected = [i for i, (a, b) in enumerate(token_spans) if a < right and b > left and b > a]
        _require(selected, "Coordinate span has no scored tokens.")
        _require(token_spans[selected[0]][0] <= left and token_spans[selected[-1]][1] >= right,
                 "Coordinate span is not completely covered by decoded tokens.")
        score = math.exp(math.fsum(logprobs[i] for i in selected) / len(selected))
        _require(math.isfinite(score) and 0 <= score <= 1, "Invalid box score.")
        ymin, xmin, ymax, xmax = obj["box_2d"]
        xyxy.append([xmin * width / 1000, ymin * height / 1000,
                     xmax * width / 1000, ymax * height / 1000])
        scores.append(score)
    return xyxy, scores


def _prediction_from_tokens(tokenizer, ids, eos_ids, max_new_tokens, logprobs, width, height):
    """Parse the complete decoded response before attempting token alignment.

    Invalid prose/Unicode and genuine [] never need coordinate-token spans.
    Recorder integrity has already been checked by predict, even for failures.
    """
    retained = ids[:-1] if ids and ids[-1] in eos_ids else ids
    text = tokenizer.decode(retained, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    truncated = len(ids) >= max_new_tokens and (not ids or ids[-1] not in eos_ids)
    result = {"pred_xyxy": [], "scores": [], "status": "format_failure",
              "generated_text": text, "generated_tokens": len(ids), "truncated": truncated}
    if truncated:
        result["error"] = "Generation reached its token limit without a stop token."
        return result
    try:
        boxes, spans = parse_answer(text)
    except (FormatFailure, SchemaFailure) as exc:
        result["status"] = "schema_failure" if isinstance(exc, SchemaFailure) else "format_failure"
        result["error"] = str(exc)
        return result
    if not boxes:
        result["status"] = "valid_empty"
        return result
    aligned_text, token_spans = decode_with_spans(tokenizer, ids, eos_ids)
    _require(aligned_text == text, "Aligned text differs from the strictly parsed response.")
    xyxy, scores = score_boxes(boxes, spans, token_spans, logprobs, width, height)
    result.update(pred_xyxy=xyxy, scores=scores, status="valid_nonempty")
    return result


def _neutral_generation_config(GenerationConfig, saved_generation, max_new_tokens,
                               tokenizer_pad_token_id=None):
    """Neutral generation with the training-time tokenizer padding override.

    The base keeps its BOS/EOS identities. Its generation PAD may be missing or
    different: training explicitly passed processor.tokenizer.pad_token_id.
    use_model_defaults=False is required. top_k is inactive for greedy decoding.
    """
    pad_id = (saved_generation.get("pad_token_id") if tokenizer_pad_token_id is None
              else tokenizer_pad_token_id)
    _require(type(pad_id) is int and pad_id >= 0, "A valid tokenizer padding token ID is required.")
    return GenerationConfig(
        max_new_tokens=max_new_tokens, do_sample=False, num_beams=1, num_return_sequences=1,
        bos_token_id=saved_generation.get("bos_token_id"), eos_token_id=saved_generation.get("eos_token_id"),
        pad_token_id=pad_id, use_cache=True,
        output_scores=False, output_logits=False, output_attentions=False,
        output_hidden_states=False, return_dict_in_generate=False,
        min_length=0, min_new_tokens=None, repetition_penalty=1.0,
        encoder_repetition_penalty=1.0, no_repeat_ngram_size=0,
        encoder_no_repeat_ngram_size=0, bad_words_ids=None, sequence_bias=None,
        forced_bos_token_id=None, forced_eos_token_id=None, suppress_tokens=None,
        begin_suppress_tokens=None, exponential_decay_length_penalty=None,
        guidance_scale=None, watermarking_config=None, renormalize_logits=False,
        remove_invalid_values=False, token_healing=False, stop_strings=None,
        disable_compile=True, temperature=1.0, top_k=50, top_p=1.0)


def _make_recorder(torch, LogitsProcessor):
    class Recorder(LogitsProcessor):
        def __init__(self):
            self.ids = []
            self.logprobs = []
            self.finite = []

        def __call__(self, input_ids, scores):
            _require(scores.ndim == 2 and scores.shape[0] == 1, "Recorder requires batch size one.")
            values, ids = scores.max(dim=-1)
            self.ids.append(ids.detach())
            self.logprobs.append((values - torch.logsumexp(scores.float(), dim=-1)).detach())
            self.finite.append(torch.isfinite(scores).all().detach())
            return scores  # no transformation or CPU synchronization

        def finish(self, actual):
            _require(len(self.ids) == len(actual), "Recorder/generated length mismatch.")
            _require(self.ids, "Generation produced no recorder steps.")
            chosen = torch.cat(self.ids).cpu().tolist()
            logprobs = torch.cat(self.logprobs).cpu().tolist()
            finite = torch.stack(self.finite).all().item()
            _require(finite, "Nonfinite raw generation logits.")
            _require(chosen == actual, "Recorder argmax does not match actual generated token IDs.")
            _require(all(math.isfinite(x) and x <= 0 for x in logprobs), "Invalid recorded logprob.")
            return logprobs
    return Recorder()


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _processor_path(run_dir, adapter_dir):
    for folder in (run_dir / "processor", run_dir / "best_adapter", adapter_dir):
        if ((folder / "preprocessor_config.json").is_file() and
                (folder / "tokenizer_config.json").is_file() and
                any((folder / name).is_file() for name in ("tokenizer.json", "tokenizer.model", "spiece.model"))):
            return folder
    raise RuntimeError("Missing local saved processor in run/processor, run/best_adapter, or selected adapter.")


def _validate_adapter(config, signature):
    _require(config.get("base_model_name_or_path") == MODEL_ID, "Unexpected adapter base model.")
    _require(config.get("revision") in (None, signature["base_revision"]), "Adapter revision mismatch.")
    _require(config.get("peft_type") == "LORA" and config.get("task_type") == "CAUSAL_LM",
             "Expected a causal-LM LoRA adapter.")
    targets = config.get("target_modules")
    # PEFT may save shortened suffixes instead of the original full paths.
    # Validate the selector here; validate its ACTUAL resolved layers below.
    if isinstance(targets, str):
        _require(targets and targets.lower() != "all-linear", "Expected an explicit LoRA target selector.")
        try:
            re.compile(targets)
        except re.error as exc:
            raise RuntimeError("Invalid adapter target regex.") from exc
    else:
        _require(isinstance(targets, list) and targets and
                 all(isinstance(name, str) and name.strip() for name in targets),
                 "Adapter target_modules must be a nonempty string list or regex.")
    _require(config.get("bias", "none") == "none" and not config.get("modules_to_save")
             and not config.get("use_dora", False), "Unexpected non-LoRA adapter parameters.")
    for key, saved in (("r", "lora_rank"), ("lora_alpha", "lora_alpha"), ("lora_dropout", "lora_dropout")):
        _require(config.get(key) == signature.get(saved), f"Adapter {key} differs from training signature.")


def _validate_resolved_targets(peft_config, named_modules, quantized_type, matches_target):
    """Use PEFT's own matching semantics, including suffixes/regex/exclusions.

    Match ALL base modules first, not only the quantized ones. Otherwise a broad
    selector could silently also target the vision tower or projector.
    """
    matched = {name: module for name, module in named_modules
               if name and matches_target(peft_config, name)}
    _require(matched, "Adapter target selector matches no base-model modules.")
    invalid = [name for name, module in matched.items()
               if "language_model" not in name.split(".")
               or name.split(".")[-1] not in PROJECTIONS
               or not isinstance(module, quantized_type)]
    _require(not invalid, "Adapter resolves outside quantized language projections: " + ", ".join(invalid[:5]))
    return sorted(matched)


class MedGemmaPredictor:
    def __init__(self, run_dir: Path, adapter_dir: Path, device: int = 0,
                 max_new_tokens: int | None = None):
        run_dir, adapter_dir = Path(run_dir), Path(adapter_dir)
        signature = json.loads((run_dir / "run_config.json").read_text())["signature"]
        _require(signature.get("model_id") == MODEL_ID, "Unexpected training base model.")
        revision = signature.get("base_revision")
        _require(isinstance(revision, str) and re.fullmatch(r"[0-9a-fA-F]{40}", revision),
                 "Training base_revision must be an exact SHA.")
        adapter_config = json.loads((adapter_dir / "adapter_config.json").read_text())
        _validate_adapter(adapter_config, signature)
        self.max_new_tokens = signature.get("max_new_tokens", 768) if max_new_tokens is None else max_new_tokens
        self.max_seq_len = signature.get("max_seq_len_guard")
        for name, value in (("max_new_tokens", self.max_new_tokens), ("max_seq_len_guard", self.max_seq_len)):
            _require(type(value) is int and value > 0, f"Invalid {name}.")
        _require(type(device) is int and device >= 0, "device must be a nonnegative CUDA index.")
        actual_versions = {name: importlib.metadata.version(name) for name in VERSIONS}
        _require(actual_versions == VERSIONS, f"Pinned dependency mismatch: expected {VERSIONS}, got {actual_versions}.")
        _require(all(signature.get("versions", {}).get(k) == v for k, v in VERSIONS.items()),
                 "Training versions do not match the supported pinned runtime.")
        import torch
        import bitsandbytes as bnb
        from peft import PeftConfig, PeftModel, prepare_model_for_kbit_training
        from peft.tuners.tuners_utils import check_target_module_exists
        from transformers import (AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig,
                                  Gemma3ForConditionalGeneration, Gemma3ImageProcessor,
                                  GenerationConfig, LogitsProcessor)
        self.torch, self.LogitsProcessor = torch, LogitsProcessor
        _require(torch.cuda.is_available() and device < torch.cuda.device_count(), "Requested CUDA device unavailable.")
        torch.cuda.set_device(device)
        _require(torch.cuda.is_bf16_supported(), "CUDA device must support BF16.")
        self.device = torch.device("cuda", device)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        quantization = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_storage=torch.uint8,
            llm_int8_skip_modules=SKIP_MODULES)
        attention = signature.get("attention")
        _require(attention in ("eager", "sdpa"), "Only the original eager/sdpa attention implementations are supported.")
        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID, revision=revision, quantization_config=quantization,
            device_map={"": device}, torch_dtype=torch.bfloat16,
            attn_implementation=attention, trust_remote_code=False, use_safetensors=True)
        _require(isinstance(model, Gemma3ForConditionalGeneration) and model.config.model_type == "gemma3",
                 "Unexpected loaded base class/architecture.")
        _require(getattr(model, "is_loaded_in_4bit", False), "Base is not loaded in 4-bit.")
        saved_generation = model.generation_config.to_dict()
        quantized = [name for name, module in model.named_modules() if isinstance(module, bnb.nn.Linear4bit)]
        _require(quantized and all(not any(part in SKIP_MODULES for part in name.split(".")) for name in quantized),
                 "Vision, projector, embeddings, or lm_head unexpectedly quantized.")
        peft_config = PeftConfig.from_pretrained(str(adapter_dir), local_files_only=True)
        resolved_targets = _validate_resolved_targets(
            peft_config, model.named_modules(), bnb.nn.Linear4bit, check_target_module_exists)
        print(f"Validated {len(resolved_targets)} resolved language-only LoRA target layers.", flush=True)
        # This deliberately upcasts frozen non-quantized parameters to FP32,
        # exactly as training did. Do not recast the model afterward.
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
        _require(all(not p.is_floating_point() or isinstance(p, bnb.nn.Params4bit) or
                     p.dtype == torch.float32 for p in model.parameters()),
                 "Frozen non-quantized parameters were not prepared in FP32 as in training.")
        model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=False,
                                          local_files_only=True, autocast_adapter_dtype=True)
        lora = [name for name, _ in model.named_parameters() if "lora_" in name]
        _require(lora and all("language_model" in name.split(".") for name in lora), "Loaded LoRA is not language-only.")
        _require(all(not p.requires_grad for p in model.parameters()), "Inference parameters unexpectedly trainable.")
        _require(all(p.device == self.device for p in model.parameters()), "Model is sharded/offloaded/on another GPU.")
        model.gradient_checkpointing_disable()
        model.eval()
        model.config.use_cache = True
        model.config.text_config.use_cache = True
        self.model = model
        processor_path = _processor_path(run_dir, adapter_dir)
        processor = AutoProcessor.from_pretrained(str(processor_path), local_files_only=True, trust_remote_code=False)
        processor.image_processor = Gemma3ImageProcessor.from_pretrained(
            str(processor_path), local_files_only=True, trust_remote_code=False)
        processor.image_processor.do_pan_and_scan = False
        processor.tokenizer.padding_side = "right"
        _require(processor.tokenizer.pad_token_id is not None, "Missing saved padding token.")
        _require(not getattr(processor.image_processor, "do_center_crop", False), "Unexpected center crop.")
        _require(hasattr(processor, "full_image_sequence") and
                 processor.image_seq_length == model.config.mm_tokens_per_image, "Unexpected image-token expansion.")
        self.processor = processor
        prompt_bytes = (run_dir / "prompt.txt").read_bytes()
        _require(hashlib.sha256(prompt_bytes).hexdigest() == signature.get("input_hashes", {}).get("prompt.txt"),
                 "Saved prompt hash differs from training.")
        prompt = prompt_bytes.decode("utf-8")
        user = {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}
        self.raw_prefix = processor.apply_chat_template([user], tokenize=False, add_generation_prompt=True)
        _require(self.raw_prefix.count(processor.boi_token) == 1, "Expected exactly one image placeholder.")
        expanded = self.raw_prefix.replace(processor.boi_token, processor.full_image_sequence)
        self.prefix_ids = processor.tokenizer(expanded, add_special_tokens=False)["input_ids"]
        eos = saved_generation.get("eos_token_id")
        self.eos_ids = eos if isinstance(eos, list) else [eos]
        _require(self.eos_ids and all(type(x) is int and x >= 0 for x in self.eos_ids), "Missing base EOS IDs.")
        # Match the original training generation call: use the saved tokenizer's
        # padding ID, not an optional/different base generation-config default.
        # Keep saved_generation unchanged for provenance and retain its EOS IDs.
        self.generation_config = _neutral_generation_config(
            GenerationConfig, saved_generation, self.max_new_tokens,
            tokenizer_pad_token_id=processor.tokenizer.pad_token_id)
        dtype_counts = {}
        for parameter in model.parameters():
            key = str(parameter.dtype)
            dtype_counts[key] = dtype_counts.get(key, 0) + parameter.numel()
        processor_hashes = {p.relative_to(processor_path).as_posix(): _hash(p)
                            for p in sorted(processor_path.rglob("*")) if p.is_file()
                            and p.name not in ("adapter_config.json", "adapter_model.safetensors")}
        self.settings = {
            "predictor": "MedGemmaPredictor", "implementation_version": 4,
            "base_model": MODEL_ID, "base_revision": revision, "model_class": type(model.get_base_model()).__name__,
            "versions": actual_versions,
            "support_versions": {name: importlib.metadata.version(name)
                                 for name in ("tokenizers", "huggingface-hub", "safetensors", "Pillow")},
            "device_index": device, "batch_size": 1,
            "max_new_tokens": self.max_new_tokens, "max_seq_len_guard": self.max_seq_len,
            "precision": "NF4 double, uint8 storage, BF16 compute/autocast; PEFT frozen FP32",
            "quantization": quantization.to_dict(), "quantization_skip_modules": SKIP_MODULES,
            "prepare_model_for_kbit_training_before_adapter": True, "autocast_adapter_dtype": True,
            "parameter_numel_by_dtype": dtype_counts, "attention": attention,
            "gradient_checkpointing": False, "use_cache": True, "inference_mode": True, "eval": True,
            "tf32_matmul": False, "cudnn_benchmark": False,
            "generation_config": self.generation_config.to_dict(), "saved_base_generation_config": saved_generation,
            "use_model_defaults": False, "synced_gpus": False, "logits_to_keep": 1,
            "logits_processors": ["RawGreedyChosenTokenLogprobRecorder"],
            "processor_class": type(processor).__name__, "image_processor_class": type(processor.image_processor).__name__,
            "tokenizer_class": type(processor.tokenizer).__name__, "processor_local_files_only": True,
            "trust_remote_code": False, "adapter_is_trainable": False,
            "adapter_config_sha256": _hash(adapter_dir / "adapter_config.json"),
            "resolved_lora_targets": resolved_targets,
            "image_processor_config": processor.image_processor.to_dict(), "processor_hashes": processor_hashes,
            "image_seq_length": processor.image_seq_length, "padding_side": "right",
            "pad_token_id_source": "Saved tokenizer, explicitly passed as in training generation",
            "do_pan_and_scan": False, "do_center_crop": False, "truncation": False,
            "prompt_sha256": hashlib.sha256(prompt_bytes).hexdigest(),
            "prompt_token_count": len(self.prefix_ids), "prompt_message_order": ["image", "text"],
            "scorer": "exp(mean(raw-selected-logit minus logsumexp(raw-vocabulary-logits)))",
            "scorer_span": "whole tokens positively overlapping the complete coordinate array, brackets included",
            "score_is_calibrated": False, "score_storage": "O(tokens) detached scalar GPU tensors; batched host transfer",
            "alignment": "valid_nonempty only: actual-ID cumulative prefix decode; cleanup=False; terminal EOS-only deletion; fail closed", 
            "parse_before_alignment": True,
            "parser": "strict full fenced JSON, duplicate keys rejected, exact fracture schema; JSONDecoder offsets",
            "hard_cap_policy": "format_failure if cap reached without terminal EOS, even if JSON parses",
            "failure_policy": "format/schema return zero predictions; runtime/alignment/scoring failures raise",
            "collection_floor": None, "box_cap": None, "nms": False,
            "model_forward_mean_ms": None,
            "model_forward_timing_note": "Unavailable: generation executes multiple decode forward calls.",
        }
        # Ensure the freeze contract really is plain JSON, not tensors/dtypes.
        self.settings = json.loads(json.dumps(self.settings, allow_nan=False))

    def predict(self, image):
        _require(getattr(image, "mode", None) == "RGB", "predict requires a PIL RGB image.")
        width, height = image.size
        _require(width > 0 and height > 0, "Empty image.")
        torch = self.torch
        inputs = self.processor(text=[self.raw_prefix], images=[[image]], padding=True,
                                truncation=False, add_special_tokens=False, return_tensors="pt",
                                images_kwargs={"do_pan_and_scan": False})
        _require(inputs["input_ids"].shape[0] == 1, "Expected one image.")
        input_length = inputs["input_ids"].shape[1]
        _require(input_length <= self.max_seq_len, "Input exceeds saved max_seq_len_guard; refusing truncation.")
        _require(inputs["input_ids"][0].tolist() == self.prefix_ids, "Processor tokenized prompt differs from training prefix.")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        recorder = _make_recorder(torch, self.LogitsProcessor)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            sequences = self.model.generate(
                **inputs, generation_config=self.generation_config, use_model_defaults=False,
                logits_processor=[recorder], synced_gpus=False, logits_to_keep=1)
        ids = sequences[0, input_length:].cpu().tolist()
        logprobs = recorder.finish(ids)
        return _prediction_from_tokens(self.processor.tokenizer, ids, self.eos_ids,
                                       self.max_new_tokens, logprobs, width, height)
