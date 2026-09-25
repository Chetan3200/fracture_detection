"""MedGemma 1.5 4B wrist-fracture SFT with language-only QLoRA.

Designed for torch 2.11.0+cu128 and one, two or four RTX 5090 GPUs.
Uses the output of prepare_medgemma.py, including its UNCHANGED saved prompt.
Never reads test images or test JSONL/reference files; optional private HF recovery backups, no trackers.

Uses Hugging Face Trainer (not TRL): the dataset is already chat-formatted.
The pinned Gemma3 implementation returns a mean masked-token loss and declares
accepts_loss_kwargs=False. With microbatch=1, ordinary Trainer accumulation/DDP
therefore optimizes an equal-image average of assistant-token mean losses.
We intentionally retain that behavior, rather than incorrectly supplying a
global token denominator that this particular model implementation ignores.

Checkpoint selection is by full validation assistant loss, NOT detection AP.
The small final generation check assesses output formatting, NOT clinical accuracy.
"""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone, timedelta
from importlib import metadata
import argparse
import copy
import csv
import hashlib
import io
import json
import math
import os
import re
import sys
import time
import shutil
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parent
MODEL_ID = "google/medgemma-1.5-4b-it"
EXPECTED_MANIFEST = "1719f37f442512c3c4fcab8350bedc0f27ff2dcebc082761299912b897747034"
VERSIONS = {"transformers": "4.57.6", "peft": "0.18.1", "accelerate": "1.12.0", "bitsandbytes": "0.49.2"}
EXPECTED_COUNTS = {
    "train": {"positive": 9479, "negative": 3961, "excluded": 803},
    "val": {"positive": 2045, "negative": 850, "excluded": 163},
    "test": {"positive": 2026, "negative": 846, "excluded": 154},
}
PROJECTIONS = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}


def require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False, default=str), encoding="utf-8")


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def parse_answer(text):
    """Strict parser. Invalid output raises; it is NEVER converted into a negative []."""
    match = re.fullmatch(r'\s*Final Answer:\s*```json\s*(.*?)\s*```\s*', text, flags=re.S)
    require(match is not None, "Missing/extra text around the required Final Answer JSON fence.")
    boxes = json.loads(match.group(1))
    require(isinstance(boxes, list), "Expected a JSON list.")
    for obj in boxes:
        require(isinstance(obj, dict) and set(obj) == {"label", "box_2d"}, "Wrong object keys.")
        require(obj["label"] == "fracture", "Wrong label.")
        box = obj["box_2d"]
        require(isinstance(box, list) and len(box) == 4, "Wrong box shape.")
        require(all(isinstance(x, (int, float)) and not isinstance(x, bool)
                    and math.isfinite(x) and 0 <= x <= 1000 for x in box), "Invalid coordinate.")
        require(box[0] < box[2] and box[1] < box[3], "Degenerate/reversed box.")
    return boxes


def manifest_targets(row):
    result = []
    boxes = json.loads(row["fracture_labels_json"])
    for cls, x, y, w, h in boxes:
        require(cls == 0, "Frozen manifest must use prepared fracture class 0.")
        corners = [y - h / 2, x - w / 2, y + h / 2, x + w / 2]
        require(all(math.isfinite(v) and -1e-6 <= v <= 1 + 1e-6 for v in corners), "Invalid frozen box bounds.")
        result.append({"box_2d": [round(min(1., max(0., v)) * 1000, 2) for v in corners], "label": "fracture"})
    return sorted(result, key=lambda obj: tuple(obj["box_2d"]))


def expected_user_message(prompt):
    return {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}


def masked_labels(input_ids, prompt_ids, attention_mask):
    """Pure, testable assistant-only label masking. Includes the assistant stop token."""
    n = len(prompt_ids)
    require(len(input_ids) == len(attention_mask), "Input/mask length mismatch.")
    require(input_ids[:n] == prompt_ids, "Tokenized chat prefix mismatch. Refusing to train on incorrectly masked labels.")
    require(all(attention_mask[:n]), "Prompt was truncated or left-padded.")
    labels = [token if index >= n and attention_mask[index] else -100
              for index, token in enumerate(input_ids)]
    require(any(v != -100 for v in labels), "No assistant tokens remain to supervise.")
    require(labels[0] == -100, "The initial token must not be supervised.")
    return labels


def stable_subset(records, count, salt):
    """Fixed image IDs independent of training seed or number of GPUs."""
    ordered = sorted(records, key=lambda r: digest((salt + "|" + r["image_id"]).encode()))
    return ordered[:min(count, len(ordered))]


def balanced_subset(records, count, categories, salt):
    positives = [r for r in records if categories[r["image_id"]] == "positive"]
    negatives = [r for r in records if categories[r["image_id"]] == "negative"]
    selected = stable_subset(positives, (count + 1) // 2, salt + "|positive")
    selected += stable_subset(negatives, count // 2, salt + "|negative")
    selected_ids = {r["image_id"] for r in selected}
    if len(selected) < min(count, len(records)):
        selected += stable_subset([r for r in records if r["image_id"] not in selected_ids],
                                  count - len(selected), salt + "|fill")
    return sorted(selected, key=lambda r: r["image_id"])


def load_bundle(data_root, expected_manifest=EXPECTED_MANIFEST):
    """Validate provenance and train/validation records without reading test examples."""
    root = Path(data_root).resolve()
    info = json.loads((root / "dataset_info.json").read_text())
    raw = (root / "split_manifest.csv").read_bytes()
    require(digest(raw) == expected_manifest == info["manifest_sha256"], "Wrong/changed frozen manifest.")
    prompt = (root / "prompt.txt").read_text(encoding="utf-8")
    require(digest(prompt.encode()) == info["prompt_sha256"], "Prompt differs from preparation metadata.")
    requirements = json.loads((root / "processor_requirements.json").read_text())
    require(requirements["model_id"] == MODEL_ID and requirements["do_pan_and_scan"] is False,
            "Unexpected model/geometry requirements.")
    rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))
    by_id = {row["filestem"]: row for row in rows}
    require(len(by_id) == len(rows), "Duplicate manifest image IDs.")
    patient_splits = {}
    for row in rows:
        patient = str(int(row["patient_id"]))
        previous = patient_splits.setdefault(patient, row["split"])
        require(previous == row["split"], "Patient leakage in frozen manifest.")
    require(Counter((r["split"], r["sample_type"]) for r in rows) ==
            Counter({(s, kind): n for s, counts in EXPECTED_COUNTS.items() for kind, n in counts.items()}),
            "Frozen split counts do not match the audited data.")
    user = expected_user_message(prompt)
    datasets = {}
    for split in ("train", "val"):
        records = read_jsonl(root / f"{split}.jsonl")
        expected_ids = {r["filestem"] for r in rows if r["split"] == split and r["sample_type"] != "excluded"}
        require(len(records) == len(expected_ids) and {r["image_id"] for r in records} == expected_ids,
                f"Missing, duplicate, excluded or extra {split} examples.")
        for record in records:
            stem = record["image_id"]
            row = by_id[stem]
            require(record["split"] == split and int(record["patient_id"]) == int(row["patient_id"]),
                    f"Record identity/split mismatch: {stem}")
            expected_image = f"images/{split}/{stem}.png"
            require(record["image"] == expected_image, f"Unexpected image mapping: {stem}")
            image = (root / record["image"]).resolve(strict=True)
            require(image.is_relative_to(root) and image.is_file(), f"Missing/unsafe image path: {stem}")
            messages = record["messages"]
            require(len(messages) == 2 and messages[0] == user and messages[1]["role"] == "assistant",
                    f"Wrong chat format or changed prompt: {stem}")
            content = messages[1]["content"]
            require(len(content) == 1 and content[0]["type"] == "text", f"Wrong answer format: {stem}")
            targets = parse_answer(content[0]["text"])
            require(targets == manifest_targets(row), f"Answer differs from frozen reference boxes: {stem}")
            require(bool(targets) == (row["sample_type"] == "positive"), f"Wrong negative target: {stem}")
        datasets[split] = records
    val_inputs = read_jsonl(root / "val_inputs.jsonl")
    require(len(val_inputs) == len(datasets["val"]) and
            {r["image_id"] for r in val_inputs} == {r["image_id"] for r in datasets["val"]},
            "Validation inference inputs do not match validation data.")
    val_by_id = {r["image_id"]: r for r in datasets["val"]}
    for item in val_inputs:
        reference = val_by_id[item["image_id"]]
        require(set(item) == {"image_id", "patient_id", "split", "image", "messages"}
                and item["messages"] == [user], "An answer/extra metadata leaked into inference inputs.")
        require(all(item[k] == reference[k] for k in ("patient_id", "split", "image")), "Inference identity mismatch.")
    filenames = ["split_manifest.csv", "prompt.txt", "train.jsonl", "val.jsonl", "val_inputs.jsonl",
                 "dataset_info.json", "processor_requirements.json"]
    return {"root": root, "prompt": prompt, "datasets": datasets, "val_inputs": val_inputs,
            "categories": {r["filestem"]: r["sample_type"] for r in rows if r["split"] in {"train", "val"}},
            "hashes": {name: file_digest(root / name) for name in filenames}, "info": info}


def validate_checkpoint(checkpoint, output, world):
    checkpoint = Path(checkpoint).resolve(strict=True)
    require(checkpoint.parent == Path(output).resolve() and checkpoint.name.startswith("checkpoint-"),
            "Resume checkpoint belongs to another run.")
    required = ["trainer_state.json", "training_args.bin", "adapter_config.json",
                "adapter_model.safetensors", "optimizer.pt", "scheduler.pt"]
    required += [f"rng_state_{r}.pth" for r in range(world)] if world > 1 else ["rng_state.pth"]
    missing = [name for name in required if not (checkpoint / name).is_file() or (checkpoint / name).stat().st_size == 0]
    require(not missing, "Incomplete resume checkpoint; missing/empty files: " + ", ".join(missing))
    state = json.loads((checkpoint / "trainer_state.json").read_text())
    require(checkpoint.name == f"checkpoint-{state['global_step']}", "Checkpoint name/optimizer step mismatch.")
    return checkpoint


SUPPORTED_WORLD_SIZES = {1, 2, 4}


def default_accumulation(world):
    require(world in SUPPORTED_WORLD_SIZES, "Supported GPU/rank counts: 1, 2, 4.")
    return 16 // world


def validate_resume_run_location(previous, output):
    require(previous.get("original_output_dir") == str(Path(output).resolve()),
            "Restore/resume into the original absolute output directory recorded in run_config.json.")


def periodic_checkpoint_due(step, interval, epoch):
    # Let the normal epoch-end callback save/evaluate at exact epoch boundaries.
    # Otherwise we could upload a pre-validation snapshot of the same step twice.
    epoch_end = epoch is not None and epoch > 0 and math.isclose(epoch, round(epoch), abs_tol=1e-8)
    return interval > 0 and step > 0 and step % interval == 0 and not epoch_end


def backup_payload(output, checkpoint, world, final=False):
    """Allowlisted, local-only recovery payload. Never include the dataset/HF cache."""
    output = Path(output).resolve()
    checkpoint = validate_checkpoint(checkpoint, output, world)
    state = json.loads((checkpoint / "trainer_state.json").read_text())
    selected = [checkpoint]
    best = state.get("best_model_checkpoint")
    if best:
        best = validate_checkpoint(best, output, world)
        if best != checkpoint:
            selected.append(best)
    required = ["run_config.json", "prompt.txt", "frozen_split_manifest.csv",
                "preprocessing_and_lora_audit.json", "training_script.py", "requirements_medgemma.txt"]
    optional = ["training_log.jsonl", "resume_events.jsonl", "hf_backup_receipts.jsonl", "resume_script.py"]
    trees = [output / "processor"] + selected
    if final:
        required += ["training_summary.json", "gpu_memory.json", "validation_format_check.jsonl",
                     "validation_format_summary.json"]
        trees.append(output / "best_adapter")
    files = []

    def include(p):
        require(not p.is_symlink(), f"Refusing to archive a symlink: {p.name}")
        require(p.resolve().is_relative_to(output), "Backup file escapes the run directory.")
        require(p.is_file(), f"Missing backup file: {p}")
        files.append(p)

    for name in required:
        include(output / name)
    for name in optional:
        if (output / name).exists():
            include(output / name)
    for folder in trees:
        require(folder.is_dir() and not folder.is_symlink(), f"Missing/unsafe backup directory: {folder}")
        require(any(folder.iterdir()), f"Empty backup directory: {folder}")
        for p in sorted(folder.rglob("*")):
            require(not p.is_symlink(), f"Refusing to archive a symlink: {p.name}")
            if p.is_file():
                include(p)
    if final:
        for name in ("adapter_config.json", "adapter_model.safetensors"):
            p = output / "best_adapter" / name
            require(p.is_file() and p.stat().st_size > 0, f"Missing final adapter file: {name}")
    return sorted(set(files)), checkpoint.name, best.name if best else None


class HFCheckpointBackup:
    """Synchronous, private, versioned recovery archives. Called by rank zero only.

    Caller must barrier AFTER all ranks save and broadcast the result/error BEFORE
    any rank resumes training. No background uploads and no adapter-only backups.
    """
    def __init__(self, api, add_operation, repo_id, output, world, retries=3, sleep_fn=time.sleep):
        self.api, self.add_operation, self.repo_id = api, add_operation, repo_id
        self.output, self.world = Path(output).resolve(), world
        self.prefix = f"runs/{self.output.name}"
        self.retries, self.sleep_fn = retries, sleep_fn
        require(retries > 0, "Upload retries must be positive.")
        require(world in SUPPORTED_WORLD_SIZES, "Unsupported backup world size.")

    def commit(self, files, message):
        last_error = None
        for attempt in range(1, self.retries + 1):
            try:
                # New operations on every attempt: upload preparation mutates them.
                operations = [self.add_operation(path_in_repo=name, path_or_fileobj=value)
                              for name, value in files]
                return self.api.create_commit(repo_id=self.repo_id, repo_type="model",
                                              operations=operations, commit_message=message)
            except Exception as exc:
                last_error = exc
                print(f"HF upload attempt {attempt}/{self.retries} failed: {type(exc).__name__}: {exc}", flush=True)
                if attempt < self.retries:
                    self.sleep_fn((15, 60)[min(attempt - 1, 1)])
        raise RuntimeError("HF backup failed after retries. Training stops; local checkpoints are retained.") from last_error

    def initialize(self, resume=False):
        self.api.whoami()
        self.api.create_repo(repo_id=self.repo_id, repo_type="model", private=True, exist_ok=True)
        require(self.api.repo_info(repo_id=self.repo_id, repo_type="model").private is True,
                "Refusing to upload to a public repository; choose a private model repo.")
        marker = f"{self.prefix}/run_config.json"
        if not resume:
            require(not self.api.file_exists(repo_id=self.repo_id, filename=marker, repo_type="model"),
                    "Remote run name already exists. Use a new --output name; do not overwrite another run.")
        # Verify WRITE permission with a small real commit before loading weights.
        result = self.commit([(marker, str(self.output / "run_config.json"))],
                             f"{self.output.name}: initialize recovery backups")
        print(f"HF write preflight passed: https://huggingface.co/{self.repo_id}", flush=True)
        return str(result.oid)

    def upload(self, checkpoint, final=False):
        require(self.api.repo_info(repo_id=self.repo_id, repo_type="model").private is True,
                "Backup repository is no longer private. Stopping without uploading.")
        files, checkpoint_name, best_name = backup_payload(self.output, checkpoint, self.world, final)
        step = int(checkpoint_name.split("-")[-1])
        kind = "final" if final else "checkpoint"
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%fZ")
        tag = f"{kind}_step_{step:06d}_{stamp}"
        remote_archive = f"{self.prefix}/backups/{tag}.zip"
        contents = {"schema_version": 1, "run_name": self.output.name,
                    "original_output_dir": str(self.output), "world_size": self.world,
                    "global_step": step, "checkpoint": checkpoint_name, "best_checkpoint": best_name,
                    "kind": kind, "created_utc": stamp,
                    "files": [p.relative_to(self.output).as_posix() for p in files],
                    "restore_policy": "Restore into the original absolute output directory; preserve all checkpoint folders."}
        needed = sum(p.stat().st_size for p in files) + 256 * 1024 * 1024
        require(shutil.disk_usage(self.output.parent).free > needed,
                "Not enough disk space to build a recovery archive; local checkpoint is intact.")
        # ZIP_STORED avoids CPU compression competing with the other GPU jobs.
        # Temp files live OUTSIDE the run and are cleaned on success or failure.
        with tempfile.TemporaryDirectory(prefix="hf-backup-", dir=self.output.parent) as tmp:
            archive = Path(tmp) / f"{tag}.zip"
            with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as z:
                for p in files:
                    z.write(p, p.relative_to(self.output).as_posix())
                z.writestr("backup_manifest.json", json.dumps(contents, indent=2))
            index = {**contents, "repo_id": self.repo_id, "archive_path": remote_archive,
                     "archive_sha256": file_digest(archive), "archive_bytes": archive.stat().st_size}
            encoded = json.dumps(index, indent=2).encode()
            result = self.commit([
                (remote_archive, str(archive)),
                (f"{self.prefix}/backups/{tag}.json", encoded),
                (f"{self.prefix}/latest.json", encoded),
            ], f"{self.output.name}: {kind} backup at optimizer step {step}")
        receipt = {**index, "commit_sha": str(result.oid), "commit_url": str(result.commit_url)}
        write_json(self.output / "hf_backup_latest.json", receipt)
        with (self.output / "hf_backup_receipts.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(receipt) + "\n")
        print(f"\nHF BACKUP COMPLETE: {kind}, optimizer step {step} -> {result.commit_url}", flush=True)
        return receipt


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=ROOT / "data/grazpedwri_medgemma")
    p.add_argument("--output", type=Path, required=True, help="New run directory; cannot overwrite an existing run")
    p.add_argument("--revision", default=None, help="Optional exact HF base-model commit; otherwise resolve main once")
    p.add_argument("--resume", type=Path, help="Trusted checkpoint-* inside the SAME output directory")
    p.add_argument("--smoke", action="store_true", help="Only 4 optimizer updates on fixed 64 train / 32 val images")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--hf-repo-id", default=os.environ.get("MEDGEMMA_HF_REPO_ID"),
                   help="Optional private model repo for full recovery archives; uses existing HF login")
    p.add_argument("--backup-steps", type=int, default=100,
                   help="Extra FULL-run checkpoint every N optimizer steps; 0 disables extras, epoch saves remain")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--grad-accum", type=int, default=None, help="Default 16/world_size; microbatch is always one image")
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--attention", choices=["sdpa", "eager"], default="sdpa")
    p.add_argument("--max-seq-len", type=int, default=2048, help="Hard safety check, NOT a truncation setting")
    p.add_argument("--generation-samples", type=int, default=16, help="Fixed balanced validation formatting check")
    p.add_argument("--max-new-tokens", type=int, default=768)
    p.add_argument("--local-rank", "--local_rank", type=int, default=None, help=argparse.SUPPRESS)
    return p.parse_args()


def main():
    args = arguments()
    require(sys.version_info >= (3, 10), "Python >=3.10 is required.")
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

    require(torch.__version__ == "2.11.0+cu128", "Use your verified torch==2.11.0+cu128 environment.")
    require(torch.cuda.is_available(), "CUDA is unavailable.")
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    env_local_rank = os.environ.get("LOCAL_RANK")
    if env_local_rank is not None and args.local_rank is not None:
        require(int(env_local_rank) == args.local_rank, "LOCAL_RANK and --local-rank disagree.")
    local_rank = int(env_local_rank) if env_local_rank is not None else (args.local_rank if args.local_rank is not None else 0)
    if world > 1:
        os.environ["LOCAL_RANK"] = str(local_rank)
    require(world in SUPPORTED_WORLD_SIZES, "This run configuration supports 1, 2, or 4 GPU/ranks.")
    require(world > 1 or torch.cuda.device_count() == 1,
            "Expose one GPU for a single process, or use torchrun with --nproc_per_node=2 or 4.")
    require(local_rank < torch.cuda.device_count(), "Rank has no matching visible GPU.")
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
                "script_sha256": file_digest(__file__), "data_root": str(bundle["root"]),
                "python": sys.version, "cuda_runtime": torch.version.cuda,
                "packages": {d.metadata["Name"]: d.version for d in metadata.distributions() if d.metadata.get("Name")},
                "test_policy": "No test image, test input JSONL, or test reference file is loaded.",
                "checkpoint_selection": "Minimum validation assistant loss, not box AP.",
                "reproducibility": "Seeds/settings/input hashes saved; GPU training is not guaranteed bitwise deterministic."}

    def initialize_output():
        if previous:
            with (output / "resume_events.jsonl").open("a") as handle:
                handle.write(json.dumps({"time": run_info["created_utc"], "checkpoint": str(checkpoint),
                                         "script_sha256": run_info["script_sha256"]}) + "\n")
        else:
            output.mkdir(parents=True, exist_ok=False)
            write_json(output / "run_config.json", run_info)
            (output / "prompt.txt").write_text(bundle["prompt"], encoding="utf-8")
            (output / "frozen_split_manifest.csv").write_bytes((bundle["root"] / "split_manifest.csv").read_bytes())
            shutil.copy2(__file__, output / "training_script.py")
            shutil.copy2(ROOT / "requirements_medgemma.txt", output / "requirements_medgemma.txt")
        if previous:
            shutil.copy2(__file__, output / "resume_script.py")
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
