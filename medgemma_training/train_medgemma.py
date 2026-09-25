"""MedGemma 1.5 4B fracture fine-tuning: the training entry point.

Flow: validate data -> initialize DDP/run -> load QLoRA model and processor ->
mask assistant labels -> train/select by validation loss -> format check/backup.
Data checks and HF archive plumbing live in medgemma_support.py.

Same recipe as the original: language-only NF4 QLoRA, microbatch 1, BF16,
equal-image assistant loss, frozen prompt/split, full-image processing without
truncation, configurable positive GPU/rank count, full checkpoint recovery, no test evaluation.
The final generation check measures formatting, NOT localization accuracy.
"""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone, timedelta
from importlib import metadata
import argparse
import json
import math
import os
import re
import shutil
import sys
import time

# Also works from a self-contained recovery archive with adjacent config/helper.
SCRIPT_ROOT = Path(__file__).resolve().parent
CONFIG_ROOT = SCRIPT_ROOT if (SCRIPT_ROOT / "config.py").is_file() else SCRIPT_ROOT.parent
for directory in (SCRIPT_ROOT, CONFIG_ROOT):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
import config as project_settings

from medgemma_support import (
    MODEL_ID, EXPECTED_MANIFEST, VERSIONS, EXPECTED_COUNTS, PROJECTIONS,
    validate_world_size, require, digest, file_digest, write_json, read_jsonl,
    parse_answer, manifest_targets, expected_user_message, masked_labels,
    stable_subset, balanced_subset, validate_checkpoint, default_accumulation,
    validate_resume_run_location, periodic_checkpoint_due, backup_payload,
    HFCheckpointBackup, load_bundle as _load_bundle,
)

ROOT = SCRIPT_ROOT


def load_bundle(data_root, expected_manifest=EXPECTED_MANIFEST):
    # Compatibility for existing callers/tests that patch the cohort constants.
    return _load_bundle(data_root, expected_manifest, expected_counts=EXPECTED_COUNTS)


def resolve_arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    defaults = project_settings.MEDGEMMA_DEFAULTS
    p.add_argument("--project-root", type=Path, help="Data/config workspace; default repository root")
    p.add_argument("--env-file", type=Path)
    p.add_argument("--data-root", type=Path, help="Default DATA_DIR/grazpedwri_medgemma")
    p.add_argument("--output", type=Path, help="Required for training: NEW run directory, or the original directory on resume")
    p.add_argument("--print-config", action="store_true", help="Offline settings preview; no model/data/GPU imports or downloads")
    p.add_argument("--revision", default=None, help="Exact base-model commit; default frozen experiment revision, or saved revision on resume")
    p.add_argument("--resume", type=Path, help="Trusted checkpoint-* inside the SAME output directory")
    p.add_argument("--smoke", action="store_true", help="Only 4 optimizer updates on fixed 64 train / 32 val images")
    p.add_argument("--epochs", type=int, default=defaults["epochs"])
    p.add_argument("--hf-repo-id",
                   help="Optional private model repo for full recovery archives; uses existing HF login")
    p.add_argument("--backup-steps", type=int, default=defaults["backup_steps"],
                   help="Extra FULL-run checkpoint every N optimizer steps; 0 disables extras, epoch saves remain")
    p.add_argument("--lr", type=float, default=defaults["lr"])
    p.add_argument("--seed", type=int, default=defaults["seed"])
    p.add_argument("--grad-accum", type=int, default=None,
                   help="MEDGEMMA_GRAD_ACCUM or exact 16/world_size; specify explicitly when GPU count does not divide 16")
    p.add_argument("--lora-rank", type=int, default=defaults["lora_rank"])
    p.add_argument("--attention", choices=["sdpa", "eager"], default=defaults["attention"])
    p.add_argument("--max-seq-len", type=int, default=defaults["max_seq_len"], help="Hard safety check, NOT a truncation setting")
    p.add_argument("--generation-samples", type=int, default=defaults["generation_samples"], help="Fixed balanced validation formatting check")
    p.add_argument("--max-new-tokens", type=int, default=defaults["max_new_tokens"])
    p.add_argument("--local-rank", "--local_rank", type=int, default=None, help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    project = project_settings.load_config(args.project_root, args.env_file)
    if args.grad_accum is None:
        configured_accum = project.setting("MEDGEMMA_GRAD_ACCUM")
        if configured_accum is not None and str(configured_accum).strip():
            args.grad_accum = int(configured_accum)
    args.data_root = project_settings.resolve_path(args.data_root or project.medgemma_data_dir, project.project_root)
    args.output = project_settings.resolve_path(args.output, project.project_root) if args.output is not None else None
    args.resume = project_settings.resolve_path(args.resume, project.project_root) if args.resume is not None else None
    require(args.output is not None or args.print_config, "Training requires --output; the Python launcher chooses a unique directory automatically.")
    if args.output is not None:
        require(not args.output.is_relative_to(args.data_root), "Put training output outside the prepared dataset.")
    if args.resume is not None:
        require(args.output is not None, "Resume requires the original --output directory.")
    elif args.revision is None:
        args.revision = project_settings.MEDGEMMA_BASE_REVISION
    if args.revision is not None:
        require(re.fullmatch(r"[a-fA-F0-9]{40}", args.revision) is not None, "--revision must be an immutable 40-character HF commit.")
        args.revision = args.revision.lower()
    args.hf_repo_id = args.hf_repo_id.strip() if args.hf_repo_id is not None else project.medgemma_hf_repo_id
    if args.hf_repo_id:
        require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*", args.hf_repo_id) is not None,
                "HF repository must be owner/repository, not a URL or credential.")
    require(args.epochs > 0 and math.isfinite(args.lr) and args.lr > 0 and args.lora_rank > 0, "Invalid training hyperparameters.")
    require(0 <= args.seed < (1 << 32) - 1, "Invalid seed.")
    require(args.backup_steps >= 0 and args.max_seq_len > 0 and args.max_new_tokens > 0 and args.generation_samples >= 0,
            "Invalid checkpoint/token/sample limits.")
    require(args.grad_accum is None or args.grad_accum > 0, "Gradient accumulation must be positive.")
    return args, project


def arguments(argv=None):
    return resolve_arguments(argv)[0]


def validate_rank_configuration(world, rank, local_rank, device_count):
    validate_world_size(world)
    require(world > 1 or device_count == 1, "Expose one GPU for a single process, or use torchrun with --nproc_per_node=N.")
    require(0 <= rank < world, "Rank is outside the configured world size.")
    require(0 <= local_rank < device_count, "Rank has no matching visible GPU.")


def validate_resume_sources(previous, script, helper, shared_config):
    require(previous.get("script_sha256") == file_digest(script),
            "Resume with the exact saved training_script.py; the executable trainer source differs.")
    for key, filename in (("helper_sha256", helper), ("config_sha256", shared_config)):
        if key in previous:
            require(previous[key] == file_digest(filename), f"Resume with the same saved {Path(filename).name} version.")


def resolved_configuration(args, project):
    world = int(project.setting("WORLD_SIZE", "1"))
    validate_world_size(world)
    accum = args.grad_accum if args.grad_accum is not None else default_accumulation(world)
    return {
        "project": project.public_dict(), "data_root": str(args.data_root),
        "output": str(args.output) if args.output is not None else None,
        "resume": str(args.resume) if args.resume is not None else None,
        "model_id": MODEL_ID, "base_revision": args.revision, "manifest_sha256": project_settings.MANIFEST_SHA256,
        "mode": "smoke" if args.smoke else "full", "world_size_from_environment": world,
        "microbatch_per_gpu": 1, "gradient_accumulation": accum, "effective_batch": world * accum,
        "training": {key: getattr(args, key) for key in project_settings.MEDGEMMA_DEFAULTS},
        "max_steps": 4 if args.smoke else -1, "hf_repo_id": args.hf_repo_id,
        "runtime_environment": {**{key: project.setting(key) for key in
                                   ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "NCCL_P2P_DISABLE", "NCCL_IB_DISABLE")},
                                "HF_HOME": str(project.hf_home) if project.hf_home is not None else None},
        "expected_versions": {**VERSIONS, "torch": project_settings.EXPECTED_TORCH},
    }


def main(argv=None):
    args, project = resolve_arguments(argv)
    if args.print_config:
        print(json.dumps(resolved_configuration(args, project), indent=2, allow_nan=False))
        return
    project.activate_environment()
    os.environ["YOLO_AUTOINSTALL"] = "false"
    run_training(args, project)


def run_training(args, project):
    require(sys.version_info >= (3, 10), "Python >=3.10 is required for training; offline configuration preview supports 3.9.")
    require(EXPECTED_MANIFEST == project_settings.MANIFEST_SHA256 and EXPECTED_COUNTS == project_settings.SPLIT_COUNTS,
            "Shared and helper frozen data contracts disagree.")
    for package, version in VERSIONS.items():
        require(metadata.version(package) == version, f"Use {package}=={version} from requirements_medgemma.txt.")
    # No tokens, credentials, environment dumps or package URLs are logged.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    import torch
    import torch.distributed as dist
    import bitsandbytes as bnb
    from PIL import Image
    from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig,
                              Gemma3ImageProcessor, Trainer, TrainerCallback, TrainingArguments, set_seed)

    require(torch.__version__ == project_settings.EXPECTED_TORCH, f"Use your verified torch=={project_settings.EXPECTED_TORCH} environment.")
    require(torch.cuda.is_available(), "CUDA is unavailable.")
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    env_local_rank = os.environ.get("LOCAL_RANK")
    if env_local_rank is not None and args.local_rank is not None:
        require(int(env_local_rank) == args.local_rank, "LOCAL_RANK and --local-rank disagree.")
    local_rank = int(env_local_rank) if env_local_rank is not None else (args.local_rank if args.local_rank is not None else 0)
    if world > 1:
        os.environ["LOCAL_RANK"] = str(local_rank)
    validate_rank_configuration(world, rank, local_rank, torch.cuda.device_count())
    torch.cuda.set_device(local_rank)
    require(torch.cuda.is_bf16_supported(), "BF16 unavailable on this GPU.")
    if world > 1:
        dist.init_process_group("nccl", timeout=timedelta(minutes=30))
    device = torch.device("cuda", local_rank)

    def primary_call(fn):
        payload = [None]
        if rank == 0:
            try:
                payload[0] = {"ok": True, "value": fn()}
            except Exception as exc:
                payload[0] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        if world > 1:
            dist.broadcast_object_list(payload, src=0, device=device)
        if not payload[0]["ok"]:
            raise RuntimeError(payload[0]["error"])
        return payload[0]["value"]

    def barrier():
        if world > 1:
            dist.barrier(device_ids=[local_rank])

    require(args.epochs > 0 and args.lr > 0 and args.lora_rank > 0, "Invalid training hyperparameters.")
    require(args.max_seq_len > 0 and args.max_new_tokens > 0 and args.generation_samples >= 0, "Invalid token/sample limits.")
    require(args.backup_steps >= 0, "--backup-steps must be nonnegative.")
    accum = args.grad_accum if args.grad_accum is not None else default_accumulation(world)
    require(accum > 0, "Gradient accumulation must be positive.")
    output = args.output.resolve()
    previous = None
    if args.resume:
        checkpoint = validate_checkpoint(args.resume, output, world)
        previous = json.loads((output / "run_config.json").read_text())
        validate_resume_run_location(previous, output)
        validate_resume_sources(previous, __file__, ROOT / "medgemma_support.py", project_settings.__file__)
        if args.hf_repo_id is None:
            args.hf_repo_id = previous.get("hf_backup", {}).get("repo_id")
        require(previous.get("run_name", output.name) == output.name, "Preserve the original run folder name on resume.")
    else:
        require(not os.path.lexists(output), "Output already exists. Use a new --output or explicitly --resume its checkpoint.")

    bundle = load_bundle(args.data_root)
    train_records = bundle["datasets"]["train"]
    val_records = bundle["datasets"]["val"]
    if args.smoke:
        train_records = balanced_subset(train_records, 64, bundle["categories"], "smoke-train-42")
        val_records = balanced_subset(val_records, 32, bundle["categories"], "smoke-val-42")
    generation_records = balanced_subset(bundle["val_inputs"], args.generation_samples,
                                         bundle["categories"], "generation-check-42")
    revision = primary_call(lambda: previous["signature"]["base_revision"] if previous else
                            HfApi().model_info(MODEL_ID, revision=args.revision or "main").sha)
    if args.resume and args.revision:
        require(args.revision == revision, "Resume base revision mismatch; omit --revision to reuse the saved commit.")
    # Catch missing model access before creating a run directory or loading weights.
    primary_call(lambda: hf_hub_download(MODEL_ID, "config.json", revision=revision))
    signature = {
        "model_id": MODEL_ID, "base_revision": revision, "input_hashes": bundle["hashes"],
        "mode": "smoke" if args.smoke else "full", "world_size": world, "microbatch_per_gpu": 1,
        "gradient_accumulation_steps": accum, "effective_batch_size": accum * world,
        "epochs": args.epochs, "max_steps": 4 if args.smoke else -1, "learning_rate": args.lr,
        "checkpoint_interval_steps": args.backup_steps,
        "checkpoint_policy": "Smoke: step 4. Full: each epoch plus optimizer-step interval without extra validation.",
        "seed": args.seed, "lora_rank": args.lora_rank, "lora_alpha": 2 * args.lora_rank,
        "lora_dropout": 0.05, "attention": args.attention, "max_seq_len_guard": args.max_seq_len,
        "optimizer": "adamw_torch_fused", "weight_decay": 0.01, "warmup_ratio": 0.03,
        "scheduler": "cosine", "max_grad_norm": 1.0, "gradient_checkpointing_reentrant": False,
        "quantization": "NF4 double quantization, uint8 storage, BF16 compute",
        "loss": "Equal-image mean of assistant-token mean cross-entropies; fixed microbatch=1",
        "train_ids": [r["image_id"] for r in train_records], "validation_ids": [r["image_id"] for r in val_records],
        "generation_ids": [r["image_id"] for r in generation_records], "max_new_tokens": args.max_new_tokens,
        "versions": {**VERSIONS, "torch": torch.__version__},
    }
    if previous:
        require(previous["signature"] == signature, "Resume settings/data/world size differ from the original run.")
    run_info = {"created_utc": datetime.now(timezone.utc).isoformat(), "signature": signature,
                "run_name": output.name, "original_output_dir": str(output),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "hf_backup": {"repo_id": args.hf_repo_id, "format": "self-contained ZIP recovery archives",
                              "extra_checkpoint_interval": args.backup_steps, "retries": 3},
                "script_sha256": file_digest(__file__), "helper_sha256": file_digest(ROOT / "medgemma_support.py"),
                "config_sha256": file_digest(project_settings.__file__),
                "configuration": resolved_configuration(args, project),
                "data_root": str(bundle["root"]),
                "python": sys.version, "python_executable": sys.executable, "cuda_runtime": torch.version.cuda,
                "packages": {d.metadata["Name"]: d.version for d in metadata.distributions() if d.metadata.get("Name")},
                "test_policy": "No test image, test input JSONL, or test reference file is loaded.",
                "checkpoint_selection": "Minimum validation assistant loss, not box AP.",
                "reproducibility": "Seeds/settings/input hashes saved; GPU training is not guaranteed bitwise deterministic."}

    def initialize_output():
        if previous:
            with (output / "resume_events.jsonl").open("a") as handle:
                handle.write(json.dumps({"time": run_info["created_utc"], "checkpoint": str(checkpoint),
                                         "script_sha256": run_info["script_sha256"],
                                         "helper_sha256": run_info["helper_sha256"],
                                         "config_sha256": run_info["config_sha256"]}) + "\n")
        else:
            output.mkdir(parents=True, exist_ok=False)
            write_json(output / "run_config.json", run_info)
            (output / "prompt.txt").write_text(bundle["prompt"], encoding="utf-8")
            (output / "frozen_split_manifest.csv").write_bytes((bundle["root"] / "split_manifest.csv").read_bytes())
            shutil.copy2(__file__, output / "training_script.py")
            shutil.copy2(ROOT / "requirements_medgemma.txt", output / "requirements_medgemma.txt")
        if previous and Path(__file__).resolve() != (output / "resume_script.py").resolve():
            shutil.copy2(__file__, output / "resume_script.py")
        config_source, config_snapshot = Path(project_settings.__file__), output / "config.py"
        if config_source.resolve() != config_snapshot.resolve():
            shutil.copy2(config_source, config_snapshot)
        require(file_digest(config_snapshot) == run_info["config_sha256"], "Shared config changed while preparing the run.")
        helper_source, helper_snapshot = ROOT / "medgemma_support.py", output / "medgemma_support.py"
        if helper_source.resolve() != helper_snapshot.resolve():
            shutil.copy2(helper_source, helper_snapshot)
        require(file_digest(helper_snapshot) == run_info["helper_sha256"],
                "Helper changed while preparing the run.")
        return str(output)

    primary_call(initialize_output)
    uploader = (HFCheckpointBackup(HfApi(), CommitOperationAdd, args.hf_repo_id, output, world)
                if args.hf_repo_id and rank == 0 else None)
    if args.hf_repo_id:
        primary_call(lambda: uploader.initialize(resume=bool(previous)))
    if rank == 0:
        print(f"Mode: {signature['mode']} | train={len(train_records):,} val={len(val_records):,}", flush=True)
        print(f"GPUs={world}, microbatch=1, accumulation={accum}, effective batch={world*accum}", flush=True)
        print("Base revision:", revision, flush=True)
        print("Test split: NOT USED", flush=True)

    set_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    # Download/load the gated model through the user's existing HF login/cache.
    # One complete replica per rank: never device_map='auto'.
    quantization = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_storage=torch.uint8,
        llm_int8_skip_modules=["vision_tower", "multi_modal_projector", "lm_head", "embed_tokens"],
    )
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, revision=revision, quantization_config=quantization, device_map={"": local_rank},
        torch_dtype=torch.bfloat16, attn_implementation=args.attention,
        trust_remote_code=False, use_safetensors=True,
    )
    require(getattr(model, "is_loaded_in_4bit", False), "Model was not loaded in 4-bit.")
    require(model.config.model_type == "gemma3", "Unexpected model architecture.")
    model.config.use_cache = False
    model.config.text_config.use_cache = False
    # Let Trainer enable non-reentrant checkpointing exactly once at train start.
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    targets = sorted(name for name, module in model.named_modules()
                     if "language_model" in name.split(".") and name.split(".")[-1] in PROJECTIONS
                     and isinstance(module, bnb.nn.Linear4bit))
    require(targets, "No quantized language-model LoRA targets found.")
    require(not any(isinstance(module, bnb.nn.Linear4bit) for name, module in model.named_modules()
                    if "vision_tower" in name.split(".") or "multi_modal_projector" in name.split(".")),
            "Vision/projector was unexpectedly quantized.")
    model = get_peft_model(model, LoraConfig(
        task_type="CAUSAL_LM", r=args.lora_rank, lora_alpha=2 * args.lora_rank,
        lora_dropout=0.05, bias="none", target_modules=targets, revision=revision,
    ))
    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    require(trainable and all("lora_" in name and "language_model" in name.split(".") for name, _ in trainable),
            "Unexpected trainable parameters outside language-model LoRA.")
    require(all(p.device == device for p in model.parameters()), "Some parameters are sharded/offloaded/on another GPU.")
    # Do not recast the whole model: PEFT deliberately leaves non-quantized base
    # parameters in FP32 for stability. Measure actual memory in the smoke run.

    processor = AutoProcessor.from_pretrained(MODEL_ID, revision=revision, trust_remote_code=False)
    # Explicitly choose the pinned slow image implementation without forcing a
    # slow tokenizer. This keeps full-image resizing reproducible.
    processor.image_processor = Gemma3ImageProcessor.from_pretrained(MODEL_ID, revision=revision)
    processor.image_processor.do_pan_and_scan = False
    processor.tokenizer.padding_side = "right"
    require(processor.tokenizer.pad_token_id is not None, "Missing padding token; do not silently modify the tokenizer.")
    require(hasattr(processor, "full_image_sequence") and processor.image_seq_length == model.config.mm_tokens_per_image,
            "Unexpected Gemma image-token expansion.")
    require(not getattr(processor.image_processor, "do_center_crop", False), "Unexpected center crop.")
    user = expected_user_message(bundle["prompt"])
    raw_prefix = processor.apply_chat_template([user], tokenize=False, add_generation_prompt=True)
    require(raw_prefix.count(processor.boi_token) == 1, "Expected exactly one full-image placeholder.")
    # Exactly mirrors pinned Gemma3Processor expansion with pan-and-scan disabled.
    expanded_prefix = raw_prefix.replace(processor.boi_token, processor.full_image_sequence)
    prefix_ids = processor.tokenizer(expanded_prefix, add_special_tokens=False)["input_ids"]

    class RecordDataset(torch.utils.data.Dataset):
        def __init__(self, records):
            self.records = records
        def __len__(self):
            return len(self.records)
        def __getitem__(self, index):
            return self.records[index]

    def image_for(record):
        with Image.open(bundle["root"] / record["image"]) as image:
            require(image.mode == "RGB", "Prepared images must already be 8-bit RGB; rerun preparation if not.")
            return image.copy()

    def collate(records):
        texts = [processor.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=False) for r in records]
        images = [[image_for(r)] for r in records]
        try:
            batch = processor(text=texts, images=images, padding=True, truncation=False,
                              add_special_tokens=False, return_tensors="pt", images_kwargs={"do_pan_and_scan": False})
        finally:
            for group in images:
                group[0].close()
        require(batch["input_ids"].shape[1] <= args.max_seq_len,
                "A complete example exceeds --max-seq-len. Increase the guard after checking VRAM; do not truncate boxes/images.")
        labels = [masked_labels(ids, prefix_ids, mask)
                  for ids, mask in zip(batch["input_ids"].tolist(), batch["attention_mask"].tolist())]
        batch["labels"] = torch.tensor(labels, dtype=torch.long)
        require("token_type_ids" in batch, "Missing multimodal token_type_ids.")
        require(torch.all(batch["labels"][batch["token_type_ids"] == 1] == -100), "Image tokens were not masked.")
        return batch

    # Check the longest tokenized target and exact processor/tokenizer prefix before
    # allocating optimizer state. No manual token truncation is ever performed.
    texts = [r["messages"][1]["content"][0]["text"] for r in train_records + val_records]
    lengths = processor.tokenizer(texts, add_special_tokens=False, return_length=True)["length"]
    require(args.max_new_tokens >= max(lengths) + 8,
            f"--max-new-tokens must be at least {max(lengths)+8} to accommodate every reference answer in this run.")
    longest_index = max(range(len(train_records)), key=lambda i: lengths[i])
    checked_batch = collate([train_records[longest_index]])
    preprocessing_audit = {
        "image_processor": type(processor.image_processor).__name__, "image_processor_config": processor.image_processor.to_dict(),
        "image_tokens": processor.image_seq_length, "prompt_token_count": len(prefix_ids),
        "checked_image_id": train_records[longest_index]["image_id"],
        "checked_input_tokens": checked_batch["input_ids"].shape[1],
        "checked_supervised_tokens": int((checked_batch["labels"] != -100).sum()),
        "maximum_answer_tokens": max(lengths), "trainable_parameters": sum(p.numel() for _, p in trainable),
        "lora_targets": targets, "trainable_parameter_names": [name for name, _ in trainable],
        "base_parameter_dtypes": dict(Counter(str(p.dtype) for p in model.parameters() if not p.requires_grad)),
    }
    del checked_batch
    if rank == 0:
        write_json(output / "preprocessing_and_lora_audit.json", preprocessing_audit)
        processor.save_pretrained(output / "processor")
        print(f"Trainable LoRA parameters: {preprocessing_audit['trainable_parameters']:,}", flush=True)
        print(f"Prompt tokens: {len(prefix_ids)}; longest answer: {max(lengths)} tokens", flush=True)

    class CheckAndLog(TrainerCallback):
        def on_step_end(self, training_args, state, control, **kwargs):
            if not args.smoke and periodic_checkpoint_due(state.global_step, args.backup_steps, state.epoch):
                control.should_save = True
            return control

        def on_save(self, training_args, state, control, **kwargs):
            if args.hf_repo_id:
                # All ranks finish writing their RNG files BEFORE rank zero reads.
                barrier()
                primary_call(lambda: uploader.upload(output / f"checkpoint-{state.global_step}"))
                # Errors are broadcast by primary_call, so all ranks fail together.
                barrier()
            return control

        def on_log(self, training_args, state, control, logs=None, **kwargs):
            if logs:
                for key in ("loss", "eval_loss", "grad_norm"):
                    if key in logs:
                        require(math.isfinite(float(logs[key])), f"Non-finite {key}; stopping rather than hiding it.")
                if state.is_world_process_zero:
                    with (output / "training_log.jsonl").open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps({"step": state.global_step, "epoch": state.epoch, **logs}, default=str) + "\n")

    training_args = TrainingArguments(
        output_dir=str(output), num_train_epochs=args.epochs, max_steps=4 if args.smoke else -1,
        per_device_train_batch_size=1, per_device_eval_batch_size=1, gradient_accumulation_steps=accum,
        learning_rate=args.lr, optim="adamw_torch_fused", weight_decay=0.01,
        lr_scheduler_type="cosine", warmup_ratio=0.03, max_grad_norm=1.0,
        bf16=True, fp16=False, tf32=False,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="steps" if args.smoke else "epoch", save_strategy="steps" if args.smoke else "epoch",
        eval_steps=4 if args.smoke else None, save_steps=4 if args.smoke else 500,
        load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
        save_total_limit=2, save_only_model=False, save_safetensors=True,
        logging_steps=1 if args.smoke else 10, logging_first_step=True, logging_nan_inf_filter=False,
        prediction_loss_only=True, remove_unused_columns=False, label_names=["labels"],
        average_tokens_across_devices=False, ddp_find_unused_parameters=False, ddp_broadcast_buffers=False,
        ddp_timeout=1800, dataloader_num_workers=0, dataloader_pin_memory=True,
        seed=args.seed, data_seed=args.seed, report_to="none", push_to_hub=False,
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=RecordDataset(train_records),
                      eval_dataset=RecordDataset(val_records), data_collator=collate,
                      processing_class=processor, callbacks=[CheckAndLog()])
    require(trainer.model_accepts_loss_kwargs is False, "Unexpected Gemma loss scaling behavior for this pinned version.")
    require(trainer.args.n_gpu == 1, "Trainer is not using exactly one GPU per rank.")
    set_seed(args.seed)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    result = trainer.train(resume_from_checkpoint=str(args.resume.resolve()) if args.resume else None)
    elapsed = time.perf_counter() - started
    require(trainer.state.best_model_checkpoint is not None, "No evaluated checkpoint was selected.")
    best_dir = output / "best_adapter"
    trainer.save_model(str(best_dir))  # Saves the selected PEFT adapter, NOT another full base model.
    barrier()
    if rank == 0:
        write_json(output / "training_summary.json", {
            "mode": signature["mode"], "train_metrics": result.metrics, "training_seconds_this_invocation": elapsed,
            "global_step": trainer.state.global_step, "best_checkpoint": trainer.state.best_model_checkpoint,
            "best_validation_loss": trainer.state.best_metric, "best_adapter": str(best_dir),
            "selection_metric": "validation assistant-token loss, not localization AP",
        })
        processor.save_pretrained(best_dir)
    gpu_stats = {"rank": rank, "gpu": torch.cuda.get_device_name(local_rank),
                 "peak_allocated_GiB": torch.cuda.max_memory_allocated(device) / 1024**3,
                 "peak_reserved_GiB": torch.cuda.max_memory_reserved(device) / 1024**3}
    all_stats = [None] * world
    if world > 1:
        dist.all_gather_object(all_stats, gpu_stats)
    else:
        all_stats[0] = gpu_stats
    if rank == 0:
        write_json(output / "gpu_memory.json", all_stats)

    def generation_check():
        # Unwrap DDP: only rank 0 generates, with no synchronization in generate().
        inference_model = trainer.accelerator.unwrap_model(trainer.model)
        inference_model.gradient_checkpointing_disable()
        inference_model.eval()
        inference_model.config.use_cache = True
        inference_model.config.text_config.use_cache = True
        valid_count = 0
        val_targets = {r["image_id"]: r["messages"][1]["content"][0]["text"] for r in bundle["datasets"]["val"]}
        eos = inference_model.generation_config.eos_token_id
        stop_ids = set(eos if isinstance(eos, list) else [eos])
        with (output / "validation_format_check.jsonl").open("w", encoding="utf-8") as handle:
            for record in generation_records:
                require(record["messages"] == [user], "Assistant answer leaked into generation.")
                image = image_for(record)
                try:
                    text = processor.apply_chat_template(record["messages"], tokenize=False, add_generation_prompt=True)
                    inputs = processor(text=[text], images=[[image]], padding=True, truncation=False,
                                       add_special_tokens=False, return_tensors="pt", images_kwargs={"do_pan_and_scan": False})
                finally:
                    image.close()
                inputs = {key: value.to(device) for key, value in inputs.items()}
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    sequences = inference_model.generate(**inputs, do_sample=False, num_beams=1,
                                                         max_new_tokens=args.max_new_tokens, use_cache=True,
                                                         synced_gpus=False, pad_token_id=processor.tokenizer.pad_token_id)
                ids = sequences[0, inputs["input_ids"].shape[1]:].tolist()
                text = processor.tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                truncated = len(ids) >= args.max_new_tokens and (not ids or ids[-1] not in stop_ids)
                parsed, error = None, None
                try:
                    require(not truncated, "Generation reached its token limit without a stop token.")
                    parsed = parse_answer(text)
                except (ValueError, TypeError, KeyError) as exc:
                    error = str(exc)
                valid_count += int(error is None)
                # Ground truth is appended AFTER generation, never supplied to generate().
                handle.write(json.dumps({"image_id": record["image_id"], "patient_id": record["patient_id"],
                                         "image": record["image"], "generated_text": text,
                                         "format_valid": error is None, "parse_error": error,
                                         "boxes": parsed, "truncated": truncated, "generated_tokens": len(ids),
                                         "reference_answer": val_targets[record["image_id"]]}, allow_nan=False) + "\n")
                handle.flush()
                print(f"Validation formatting check: {valid_count} valid so far", flush=True)
        summary = {"images": len(generation_records), "format_valid": valid_count,
                   "format_valid_fraction": valid_count / len(generation_records) if generation_records else None,
                   "sample_policy": "Fixed balanced validation subset, not representative performance evaluation",
                   "note": "No box AP or calibrated confidence scores are computed here. Invalid outputs are not negatives."}
        write_json(output / "validation_format_summary.json", summary)
        return summary

    primary_call(generation_check)
    if args.hf_repo_id:
        barrier()
        primary_call(lambda: uploader.upload(output / f"checkpoint-{trainer.state.global_step}", final=True))
        barrier()
    if rank == 0:
        print("\nFinished:", output, flush=True)
        print("Selected adapter:", best_dir, flush=True)
        print("Selection used validation loss only. Test split remains unused.", flush=True)
        if args.smoke:
            print("SMOKE RUN ONLY: do not use this adapter as the final model or resume it into the full experiment.", flush=True)
    barrier()
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
