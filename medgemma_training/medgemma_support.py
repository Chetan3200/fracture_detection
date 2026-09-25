"""Data checks, answer parsing/masking, and private HF recovery archives.

Standard-library only: safe to import for evaluation utilities and CPU tests.
The training/model code lives in train_medgemma.py.
"""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import csv
import hashlib
import io
import json
import math
import re
import shutil
import tempfile
import time
import zipfile

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


def load_bundle(data_root, expected_manifest=EXPECTED_MANIFEST, expected_counts=None):
    """Validate provenance and train/validation records without reading test examples."""
    expected_counts = EXPECTED_COUNTS if expected_counts is None else expected_counts
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
            Counter({(s, kind): n for s, counts in expected_counts.items() for kind, n in counts.items()}),
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


def validate_world_size(world):
    if type(world) is not int or world <= 0:
        raise ValueError(f"World size must be a positive integer GPU/rank count; got {world!r}.")
    return world


def validate_checkpoint(checkpoint, output, world):
    validate_world_size(world)
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


def default_accumulation(world):
    validate_world_size(world)
    if 16 % world:
        raise ValueError(
            f"World size {world} does not divide effective batch 16; "
            "set --grad-accum explicitly (batch = world_size * grad_accum)."
        )
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
    # New runs need the helper in their recovery archive. Legacy monolithic runs
    # remain supported; a resumed legacy run may now also have a helper snapshot.
    run_info = json.loads((output / "run_config.json").read_text())
    helper = output / "medgemma_support.py"
    if run_info.get("helper_sha256"):
        require(helper.is_file() and not helper.is_symlink() and file_digest(helper) == run_info["helper_sha256"],
                "Training helper is missing or changed; refusing an incomplete recovery archive.")
        required.append("medgemma_support.py")
    elif helper.exists():
        optional.append("medgemma_support.py")
    config = output / "config.py"
    if "config_sha256" in run_info:
        require(config.is_file() and not config.is_symlink() and file_digest(config) == run_info["config_sha256"],
                "Training config is missing or changed; refusing an incomplete recovery archive.")
        required.append("config.py")
    elif config.exists():
        optional.append("config.py")
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
        validate_world_size(world)

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


