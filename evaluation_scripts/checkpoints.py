"""Resolve local or immutable Hugging Face checkpoints without loading any model.

Only the standard library is imported here. Install huggingface_hub for remote
resolution. ZIP restoration is for inference, not training resume; original
absolute output paths are deliberately not required.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tempfile
import zipfile

DEFAULT_YOLO_REPO = "Crimson-Dawn/grazpedwri-yolo26-checkpoints"
DEFAULT_MEDGEMMA_REPO = "Crimson-Dawn/medgemma-fracture-checkpoints"
EXPECTED_MANIFEST_SHA256 = "1719f37f442512c3c4fcab8350bedc0f27ff2dcebc082761299912b897747034"
MAX_SELECTED_BYTES = 4 * 1024**3
MAX_METADATA_BYTES = 16 * 1024**2
_CACHE_MANIFEST = ".content_manifest.json"
_METADATA = frozenset({"run_config.json", "prompt.txt", "frozen_split_manifest.csv",
                       "training_summary.json", "preprocessing_and_lora_audit.json"})
_COMMIT = re.compile(r"[0-9a-fA-F]{40}\Z")
_SHA256 = re.compile(r"[0-9a-fA-F]{64}\Z")
_CHECKPOINT = re.compile(r"checkpoint-[0-9]+\Z")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256_file(path):
    """Hash a file using bounded memory, without deserializing it."""
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _json(path):
    path = Path(path)
    _require(path.is_file(), f"Missing JSON file: {path}")
    _require(path.stat().st_size <= MAX_METADATA_BYTES, f"Metadata too large: {path.name}")
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    _require(isinstance(value, dict), f"Expected a JSON object: {path.name}")
    return value


def _cache(cache_dir):
    root = Path(cache_dir).expanduser() if cache_dir else Path.home() / ".cache" / "fracture_evaluation"
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def _remote(repo, revision, cache_dir):
    """Import HF lazily and resolve the requested ref once, before any download."""
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as exc:
        raise ImportError("Remote checkpoints require huggingface_hub; install it first.") from exc
    api = HfApi()
    commit = api.model_info(repo_id=repo, revision=revision or "main").sha
    _require(isinstance(commit, str) and _COMMIT.fullmatch(commit),
             "HF did not return an exact 40-hex model commit.")
    commit = commit.lower()
    cache = _cache(cache_dir)

    def download(filename):
        _safe_name(filename)
        return Path(hf_hub_download(repo_id=repo, filename=filename,
                                   revision=commit, cache_dir=str(cache / "huggingface")))
    return api, commit, download


def resolve_yolo(local=None, repo=None, filename=None, revision=None, cache_dir=None):
    """Return (PT path, provenance). HF filenames must be explicit repo paths.

    Pass local OR remote arguments, never both. No torch/pickle data is loaded.
    """
    if local is not None:
        _require(repo is None and filename is None and revision is None,
                 "Choose local YOLO OR HF repo/filename/revision, not both.")
        # Preserve the .pt suffix even when this is an HF-cache symlink to an
        # extensionless blob: Ultralytics uses the supplied filename's suffix.
        weights = Path(local).expanduser().absolute()
        source = {"source": "local", "resolved_revision": None}
    else:
        _require(isinstance(filename, str) and filename, "HF YOLO requires an explicit --hf-filename.")
        _safe_name(filename)
        _require(PurePosixPath(filename).suffix.lower() == ".pt", "YOLO filename must end in .pt.")
        repo = repo or DEFAULT_YOLO_REPO
        _, commit, download = _remote(repo, revision, cache_dir)
        weights = download(filename)
        source = {"source": "huggingface", "repo_id": repo, "filename": filename,
                  "requested_revision": revision or "main", "resolved_revision": commit}
    # HF cache symlinks may point at an extensionless blob. Check the supplied
    # filename, not the resolved blob basename.
    _require(weights.is_file() and weights.stat().st_size > 0, f"Missing/empty YOLO weights: {weights}")
    if local is not None:
        _require(Path(local).suffix.lower() == ".pt", "Local YOLO weights must be a .pt file.")
    checksum = sha256_file(weights)
    return weights, {**source, "weights_path": str(weights), "weights_sha256": checksum}


def _safe_name(name):
    """Reject traversal, ambiguous separators, Windows paths and NUL bytes."""
    _require(isinstance(name, str) and name and "\\" not in name and "\x00" not in name,
             "Unsafe archive/repository path.")
    _require(not name.startswith("/") and ":" not in name.split("/")[0],
             f"Absolute or drive-qualified path rejected: {name!r}")
    clean = name[:-1] if name.endswith("/") else name
    _require(clean and all(part not in ("", ".", "..") for part in clean.split("/")),
             f"Noncanonical/traversing path rejected: {name!r}")
    return PurePosixPath(clean).as_posix()


def _selected(name):
    return name in _METADATA or name.startswith("best_adapter/") or name.startswith("processor/")


def _zip_files(archive):
    """Validate EVERY member, including members that will not be extracted."""
    seen, files, directories = set(), {}, set()
    total = 0
    for entry in archive.infolist():
        original = getattr(entry, "orig_filename", entry.filename)
        name = _safe_name(original)
        _require(name not in seen, f"Duplicate normalized ZIP entry: {name}")
        seen.add(name)
        mode = entry.external_attr >> 16
        kind = stat.S_IFMT(mode)
        _require(kind in (0, stat.S_IFREG, stat.S_IFDIR), f"Symlink/special ZIP member rejected: {name}")
        _require(not (kind == stat.S_IFDIR and not entry.is_dir()) and
                 not (kind == stat.S_IFREG and entry.is_dir()), f"Conflicting ZIP member type: {name}")
        _require(not entry.flag_bits & 1, f"Encrypted ZIP member rejected: {name}")
        if entry.is_dir():
            directories.add(name)
        else:
            files[name] = entry
            if _selected(name):
                total += entry.file_size
                _require(total <= MAX_SELECTED_BYTES, "Selected ZIP contents exceed the 4 GiB safety limit.")
    for name in seen:
        for parent in PurePosixPath(name).parents:
            if str(parent) != ".":
                _require(str(parent) not in files, f"ZIP file/directory collision: {parent}")
    chosen = {name: entry for name, entry in files.items() if _selected(name)}
    _require(chosen, "Archive has no allowlisted inference files.")
    return chosen


def _verify_extraction(target, archive_sha, selected):
    """Check the complete cached file set and every content hash, not a flag."""
    _require(target.is_dir() and not target.is_symlink(), f"Unsafe extraction cache: {target}")
    present = set()
    for item in target.rglob("*"):
        _require(not item.is_symlink(), f"Symlink in extraction cache: {item}")
        _require(item.is_file() or item.is_dir(), f"Special file in extraction cache: {item}")
        if item.is_file():
            present.add(item.relative_to(target).as_posix())
    _require(present == set(selected) | {_CACHE_MANIFEST},
             f"Extraction cache file set changed; remove this cache directory and retry: {target}")
    manifest = _json(target / _CACHE_MANIFEST)
    _require(manifest.get("schema_version") == 1 and manifest.get("archive_sha256") == archive_sha,
             "Extraction cache belongs to a different archive/version.")
    records = manifest.get("files")
    _require(isinstance(records, dict) and set(records) == set(selected), "Invalid extraction content manifest.")
    for name, entry in selected.items():
        record = records[name]
        item = target / name
        _require(isinstance(record, dict) and record.get("size") == entry.file_size and
                 record.get("crc32") == entry.CRC and item.stat().st_size == entry.file_size,
                 f"Extraction cache size/CRC metadata mismatch: {name}")
        _require(sha256_file(item) == record.get("sha256"),
                 f"Extraction cache hash mismatch: {name}; remove {target} and retry.")


def _extract(archive_path, cache_dir, expected_sha=None):
    """Stream only inference files to an atomic, content-addressed cache."""
    actual_sha = sha256_file(archive_path)
    if expected_sha is not None:
        _require(isinstance(expected_sha, str) and _SHA256.fullmatch(expected_sha), "Invalid archive SHA256.")
        _require(actual_sha == expected_sha.lower(), "Archive SHA256 does not match latest.json.")
    parent = _cache(cache_dir) / "medgemma_extracted"
    parent.mkdir(exist_ok=True)
    _require(not parent.is_symlink(), "Extraction root cannot be a symlink.")
    target = parent / actual_sha
    with zipfile.ZipFile(archive_path) as archive:
        selected = _zip_files(archive)
        if target.exists() or target.is_symlink():
            _verify_extraction(target, actual_sha, selected)
            return target, actual_sha
        stage = Path(tempfile.mkdtemp(prefix=f".{actual_sha}.", dir=parent))
        try:
            records = {}
            total_written = 0
            for name, entry in selected.items():
                dest = stage / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                digest, written = hashlib.sha256(), 0
                # Reading ZipExtFile to EOF verifies this member's ZIP CRC.
                with archive.open(entry) as source, dest.open("xb") as output:
                    while True:
                        block = source.read(1024 * 1024)
                        if not block:
                            break
                        written += len(block)
                        total_written += len(block)
                        _require(written <= entry.file_size and total_written <= MAX_SELECTED_BYTES,
                                 "ZIP extraction size limit exceeded.")
                        digest.update(block)
                        output.write(block)
                _require(written == entry.file_size, f"Truncated archive member: {name}")
                records[name] = {"sha256": digest.hexdigest(), "size": written, "crc32": entry.CRC}
            (stage / _CACHE_MANIFEST).write_text(json.dumps({"schema_version": 1,
                "archive_sha256": actual_sha, "files": records}, indent=2), encoding="utf-8")
            try:
                stage.rename(target)
            except OSError:
                if not target.exists():
                    raise
                # Another resolver may have published the same archive first.
                _verify_extraction(target, actual_sha, selected)
            _verify_extraction(target, actual_sha, selected)
        finally:
            if stage.exists():
                shutil.rmtree(stage)
    return target, actual_sha


def _run_name(run):
    _require(isinstance(run, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run),
             "--hf-run must be one safe run directory name, not a path.")
    return run


def _pointer(marker, run):
    _require(marker.get("kind") == "final", f"Run {run} is not a completed final backup.")
    _require(marker.get("run_name") == run, f"Run name mismatch in latest.json for {run}.")
    archive = marker.get("archive_path")
    _safe_name(archive)
    _require(re.fullmatch(rf"runs/{re.escape(run)}/backups/final_step_[^/]+\.zip", archive),
             f"Unexpected final archive path for {run}.")
    checksum = marker.get("archive_sha256")
    _require(isinstance(checksum, str) and _SHA256.fullmatch(checksum), f"Invalid archive checksum for {run}.")
    for key in ("checkpoint", "best_checkpoint"):
        _require(isinstance(marker.get(key), str) and _CHECKPOINT.fullmatch(marker[key]),
                 f"Missing/invalid {key} in final marker for {run}.")
    return archive, checksum.lower()


def _remote_run(api, download, repo, commit, run):
    """Read only latest pointers and run configs; never guess the latest run."""
    explicit = run is not None
    if explicit:
        names = [_run_name(run)]
    else:
        paths = api.list_repo_files(repo_id=repo, revision=commit, repo_type="model")
        names = sorted({match.group(1) for name in paths
                        if (match := re.fullmatch(r"runs/([^/]+)/latest\.json", name))})
    eligible, rejected = [], []
    for name in names:
        try:
            _run_name(name)
            marker = _json(download(f"runs/{name}/latest.json"))
            archive, checksum = _pointer(marker, name)
            config = _json(download(f"runs/{name}/run_config.json"))
            signature = config.get("signature", {})
            _require(isinstance(signature, dict) and signature.get("mode") == "full",
                     f"Run {name} is not mode=full.")
            _require(config.get("run_name") == name, f"Run config name mismatch for {name}.")
            eligible.append((name, marker, config, archive, checksum))
        except (ValueError, FileNotFoundError) as exc:
            if explicit:
                raise
            rejected.append(f"{name}: {exc}")
    _require(len(eligible) == 1,
             "Specify --hf-run; eligible full completed runs: " +
             (", ".join(item[0] for item in eligible) or "(none)") +
             (". Ineligible: " + "; ".join(rejected) if rejected else ""))
    return eligible[0]


def _nonempty_file(path):
    _require(path.is_file() and not path.is_symlink() and path.stat().st_size > 0,
             f"Missing/empty/unsafe required file: {path}")


def _validate_medgemma(run_dir, adapter_dir, selection, marker=None, remote_config=None):
    """Verify frozen inputs and base identity; do not deserialize adapter weights."""
    for name in ("run_config.json", "prompt.txt", "frozen_split_manifest.csv"):
        _nonempty_file(run_dir / name)
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        _nonempty_file(adapter_dir / name)
    config = _json(run_dir / "run_config.json")
    if remote_config is not None:
        _require(config == remote_config, "Archive run_config differs from the commit-pinned remote config.")
    signature = config.get("signature")
    _require(isinstance(signature, dict), "run_config.json is missing signature.")
    _require(signature.get("mode") == "full", "Evaluation requires a full training run, not smoke.")
    model = signature.get("model_id")
    revision = signature.get("base_revision")
    _require(isinstance(model, str) and model.strip(), "Missing base model ID.")
    _require(isinstance(revision, str) and _COMMIT.fullmatch(revision),
             "Base revision must be an exact 40-hex commit in run_config.signature.")
    hashes = signature.get("input_hashes")
    _require(isinstance(hashes, dict), "Missing run_config.signature.input_hashes.")
    manifest_sha = sha256_file(run_dir / "frozen_split_manifest.csv")
    _require(manifest_sha == EXPECTED_MANIFEST_SHA256, "Frozen split manifest is not the expected evaluation manifest.")
    _require(hashes.get("split_manifest.csv") == manifest_sha, "Training split-manifest input hash mismatch.")
    prompt_sha = sha256_file(run_dir / "prompt.txt")
    _require(hashes.get("prompt.txt") == prompt_sha, "Training prompt input hash mismatch.")
    adapter_config = _json(adapter_dir / "adapter_config.json")
    _require(adapter_config.get("base_model_name_or_path") == model, "Adapter/run-config base model mismatch.")
    # PEFT can leave revision null. The authoritative pinned revision is saved
    # in run_config; if the adapter also records a revision, it must agree.
    if adapter_config.get("revision") is not None:
        _require(adapter_config["revision"] == revision, "Adapter/run-config base revision mismatch.")
    def has_processor(folder):
        return ((folder / "preprocessor_config.json").is_file() and
                (folder / "tokenizer_config.json").is_file() and
                any((folder / name).is_file() for name in
                    ("tokenizer.json", "tokenizer.model", "spiece.model")))

    processor = next((folder for folder in
                      (run_dir / "processor", run_dir / "best_adapter", adapter_dir)
                      if has_processor(folder)), None)
    _require(processor is not None,
             "Missing saved processor/tokenizer files in run/processor or best_adapter.")
    _require(not processor.is_symlink(), "Processor directory cannot be a symlink.")
    processor_hashes = {}
    for item in sorted(processor.rglob("*")):
        _require(not item.is_symlink(), f"Symlink in processor files: {item}")
        if item.is_file() and item.name not in ("adapter_config.json", "adapter_model.safetensors"):
            processor_hashes[item.relative_to(processor).as_posix()] = sha256_file(item)
    summary_path = run_dir / "training_summary.json"
    summary = _json(summary_path) if summary_path.exists() else None
    best = None
    if summary is not None:
        _require(summary.get("mode") == "full", "training_summary mode disagrees with full run.")
        best_path = summary.get("best_checkpoint")
        _require(isinstance(best_path, str) and _CHECKPOINT.fullmatch(PurePosixPath(best_path).name),
                 "training_summary has no valid selected best checkpoint.")
        best = PurePosixPath(best_path).name
        _require(isinstance(summary.get("best_adapter"), str) and
                 PurePosixPath(summary["best_adapter"]).name == "best_adapter",
                 "training_summary does not identify best_adapter.")
        metric = summary.get("selection_metric")
        _require(isinstance(metric, str) and "loss" in metric.lower() and "validation" in metric.lower(),
                 "Unexpected/missing training checkpoint selection metric.")
        if marker is not None:
            _require(best == marker["best_checkpoint"], "Marker/summary best checkpoint mismatch.")
    elif marker is not None:
        raise ValueError("Final HF backup is missing training_summary.json.")
    if marker is not None:
        _require(config.get("run_name") == marker["run_name"], "Marker/archive run name mismatch.")
    note = ("Explicit local checkpoint, not automatically chosen as best." if selection == "explicit_checkpoint"
            else "Saved best_adapter; training selected minimum validation assistant-token loss, not localization AP.")
    return {"run_dir": str(run_dir), "adapter_dir": str(adapter_dir), "processor_dir": str(processor),
            "run_name": config.get("run_name"), "selection": selection, "selection_note": note,
            "best_checkpoint": best, "training_summary_present": summary is not None,
            "training_summary_sha256": sha256_file(summary_path) if summary is not None else None,
            "base_model": model, "base_revision": revision.lower(),
            "weights_path": str(adapter_dir / "adapter_model.safetensors"),
            "weights_sha256": sha256_file(adapter_dir / "adapter_model.safetensors"),
            "config_sha256": sha256_file(adapter_dir / "adapter_config.json"),
            "run_config_sha256": sha256_file(run_dir / "run_config.json"),
            "manifest_sha256": manifest_sha, "prompt_sha256": prompt_sha,
            "processor_sha256": processor_hashes}


def resolve_medgemma(local=None, repo=None, run=None, revision=None, cache_dir=None):
    """Return (run_dir, adapter_dir, provenance) for inference only.

    Local input may be a full run, its best_adapter, an explicit checkpoint-N,
    or a full ZIP backup. Direct adapter/checkpoint paths need their original
    parent run metadata (but not its original absolute location). Remote input
    must be a final, full backup; ambiguity always requires an explicit run.
    """
    marker = remote_config = None
    selection = "best_adapter"
    if local is not None:
        _require(repo is None and run is None and revision is None,
                 "Choose local MedGemma OR HF repo/run/revision, not both.")
        path = Path(local).expanduser().resolve()
        provenance = {"source": "local", "resolved_revision": None, "local_path": str(path)}
        if path.is_file():
            _require(path.suffix.lower() == ".zip", "Local MedGemma file must be a full backup ZIP.")
            run_dir, checksum = _extract(path, cache_dir)
            adapter_dir = run_dir / "best_adapter"
            provenance.update(source="local_zip", archive_path=str(path), archive_sha256=checksum)
        elif path.is_dir():
            if path.name == "best_adapter" or _CHECKPOINT.fullmatch(path.name):
                adapter_dir, run_dir = path, path.parent
                if _CHECKPOINT.fullmatch(path.name):
                    selection = "explicit_checkpoint"
            else:
                run_dir, adapter_dir = path, path / "best_adapter"
        else:
            raise ValueError(f"Local MedGemma path does not exist: {path}")
    else:
        repo = repo or DEFAULT_MEDGEMMA_REPO
        api, commit, download = _remote(repo, revision, cache_dir)
        name, marker, remote_config, archive_name, checksum = _remote_run(api, download, repo, commit, run)
        archive_path = download(archive_name)
        run_dir, checksum = _extract(archive_path, cache_dir, checksum)
        adapter_dir = run_dir / "best_adapter"
        provenance = {"source": "huggingface", "repo_id": repo, "hf_run": name,
                      "requested_revision": revision or "main", "resolved_revision": commit,
                      "archive_filename": archive_name, "archive_path": str(archive_path),
                      "archive_sha256": checksum, "marker_checkpoint": marker["checkpoint"],
                      "marker_best_checkpoint": marker["best_checkpoint"]}
    _require(adapter_dir.is_dir() and not adapter_dir.is_symlink(), f"Missing/unsafe adapter directory: {adapter_dir}")
    details = _validate_medgemma(run_dir, adapter_dir, selection, marker, remote_config)
    return run_dir, adapter_dir, {**provenance, **details}
