"""Back up frozen YOLO + MedGemma data as private, numbered TAR/JSON parts.

CLI > exported environment > .env > config.py. Archive paths remain portable
regardless of DATA_DIR. --help / --print-config are offline and stdlib-only.
No image transformations or new train/val/test split are performed.
"""
import argparse
from collections import Counter
from dataclasses import replace
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import tarfile
import tempfile

import config

DATA_FOLDERS = config.DATA_ARCHIVE_FOLDERS
METADATA_TYPES = {".txt", ".json", ".jsonl", ".csv", ".yaml", ".yml", ".md"}
file_hash = config.sha256


def json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def save_json(path, value):
    Path(path).write_bytes(json_bytes(value))


def resolve_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", help="Private dataset repository; default DATASET_HF_REPO_ID")
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--manifest-path", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--shard-mib", type=int, default=1024)
    parser.add_argument("--print-config", action="store_true")
    args = parser.parse_args(argv)
    project = config.load_config(args.project_root, args.env_file)
    overrides = {name: config.resolve_path(getattr(args, name), project.project_root)
                 for name in ("data_dir", "manifest_path", "cache_dir") if getattr(args, name) is not None}
    project = replace(project, **overrides)
    args.repo_id = args.repo_id.strip() if args.repo_id is not None else project.dataset_hf_repo_id
    if args.shard_mib < 1:
        raise ValueError("--shard-mib must be positive.")
    if args.repo_id and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*", args.repo_id) is None:
        raise ValueError("Dataset repository must be owner/repository, not a URL or credential.")
    if not args.repo_id and not args.print_config:
        raise ValueError("Set DATASET_HF_REPO_ID in .env/environment or pass --repo-id.")
    for folder in config.dataset_directories(project).values():
        if project.cache_dir.is_relative_to(folder):
            raise ValueError("Put CACHE_DIR outside the prepared datasets.")
    return args, project


def manifest_rows(project):
    manifest_bytes = project.manifest_path.read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != config.MANIFEST_SHA256:
        raise ValueError(f"Frozen manifest checksum mismatch: {project.manifest_path}")
    for folder in config.dataset_directories(project).values():
        if file_hash(folder / "split_manifest.csv") != config.MANIFEST_SHA256:
            raise ValueError(f"Frozen manifest checksum mismatch: {folder}")
    # Parse the same bytes that were hashed, not a second potentially changed read.
    with io.StringIO(manifest_bytes.decode("utf-8-sig"), newline="") as handle:
        return [r for r in csv.DictReader(handle) if r["sample_type"] != "excluded"]


def list_files(project, rows):
    """Map canonical archive names to physical files, including dereferenced images."""
    files = {"split_manifest.csv": project.manifest_path}
    for logical, folder in config.dataset_directories(project).items():
        for path in folder.rglob("*"):
            parts = path.relative_to(folder).parts
            if any(p.startswith(".") or p in {"images", "cache", "__pycache__", "runs", "checkpoints"} for p in parts):
                continue
            if path.is_file() and path.suffix.lower() in METADATA_TYPES:
                if not path.resolve().is_relative_to(folder.resolve()):
                    raise ValueError(f"Metadata link leaves its dataset folder: {path}")
                files[f"{logical}/{path.relative_to(folder).as_posix()}"] = path
        for row in rows:
            # Original YOLO bytes keep their original extension. MedGemma is RGB PNG.
            suffix = Path(row["image_relpath"]).suffix if logical == DATA_FOLDERS[0] else ".png"
            name = f"images/{row['split']}/{row['filestem']}{suffix}"
            files[f"{logical}/{name}"] = folder / name
    for row in rows:
        name = f"labels/{row['split']}/{row['filestem']}.txt"
        files[f"{DATA_FOLDERS[0]}/{name}"] = project.yolo_data_dir / name
    for name, path in files.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing prepared dataset file: {path}")
    if f"{DATA_FOLDERS[0]}/data.yaml" not in files:
        raise FileNotFoundError("Prepared YOLO data.yaml is missing.")
    return dict(sorted(files.items()))


def group_files(records, byte_limit):
    groups, current, current_size = [], [], 0
    for record in records:
        size = record["size"]
        if size > byte_limit:
            raise ValueError(f"Increase --shard-mib: file is too large: {record['path']}")
        if current and current_size + size > byte_limit:
            groups.append(current)
            current, current_size = [], 0
        current.append(record)
        current_size += size
    if current:
        groups.append(current)
    return groups


def upload_files(api, repo_id, files, message):
    from huggingface_hub import CommitOperationAdd
    if api.repo_info(repo_id, repo_type="dataset").private is not True:
        raise ValueError("Use a private Hugging Face dataset repository.")
    return api.create_commit(repo_id=repo_id, repo_type="dataset", commit_message=message,
        operations=[CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(path)) for name, path in files.items()])


def main(argv=None):
    args, project = resolve_arguments(argv)
    if args.print_config:
        print(json.dumps({"action": "backup", "project": project.public_dict(), "repo_id": args.repo_id,
                          "datasets": {k: str(v) for k, v in config.dataset_directories(project).items()},
                          "manifest_sha256": config.MANIFEST_SHA256, "shard_mib": args.shard_mib}, indent=2))
        return
    rows = manifest_rows(project)
    files = list_files(project, rows)
    records = []
    print(f"Indexing {len(files):,} files...", flush=True)
    for number, (name, path) in enumerate(files.items(), 1):
        records.append({"path": name, "size": path.stat().st_size, "sha256": file_hash(path)})
        if number % 2000 == 0:
            print(f"Indexed {number:,}/{len(files):,}", flush=True)
    manifest_names = {"split_manifest.csv"} | {f"{f}/split_manifest.csv" for f in DATA_FOLDERS}
    if any(r["sha256"] != config.MANIFEST_SHA256 for r in records if r["path"] in manifest_names):
        raise ValueError("A frozen manifest changed during indexing.")
    parts = group_files(records, args.shard_mib * 1024 * 1024)
    inventory = {"format_version": 2, "manifest_sha256": config.MANIFEST_SHA256,
                 "shard_mib": args.shard_mib, "files": records}
    snapshot_id = hashlib.sha256(json_bytes(inventory)).hexdigest()
    prefix = f"snapshots/{snapshot_id}"

    # All local validation/indexing precedes any Hub operation. Never store tokens.
    project.activate_environment()
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(repo_id=args.repo_id, repo_type="dataset", private=True, exist_ok=True)
    info = api.repo_info(args.repo_id, repo_type="dataset")
    if info.private is not True:
        raise ValueError("Use a private Hugging Face dataset repository.")
    if info.sha is not None:
        if re.fullmatch(r"[a-f0-9]{40}", info.sha) is None:
            raise ValueError("Hub did not return an immutable commit ID.")
        if api.file_exists(args.repo_id, f"{prefix}/BACKUP_COMPLETE.json", repo_type="dataset", revision=info.sha):
            print(f"Snapshot already complete: {snapshot_id}\nImmutable HF revision: {info.sha}", flush=True)
            return

    project.cache_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="dataset_upload_", dir=project.cache_dir) as temporary:
        work = Path(temporary)
        save_json(work / "FILE_INVENTORY.json", inventory)
        save_json(work / "UPLOAD_STARTED.json", {"status": "uploading", "snapshot_id": snapshot_id,
                  "expected_shards": len(parts), "file_count": len(records)})
        upload_files(api, args.repo_id, {f"{prefix}/FILE_INVENTORY.json": work / "FILE_INVENTORY.json",
                     f"{prefix}/UPLOAD_STARTED.json": work / "UPLOAD_STARTED.json"}, "Start prepared dataset backup")
        receipts = []
        for number, part in enumerate(parts, 1):
            name = f"part-{number:05d}"
            archive_path = work / f"{name}.tar"
            print(f"Creating/uploading part {number}/{len(parts)}...", flush=True)
            with tarfile.open(archive_path, "w", format=tarfile.PAX_FORMAT) as archive:
                for record in part:
                    content = files[record["path"]].read_bytes()
                    if len(content) != record["size"] or hashlib.sha256(content).hexdigest() != record["sha256"]:
                        raise ValueError(f"Dataset file changed during backup: {record['path']}")
                    entry = tarfile.TarInfo(record["path"])
                    entry.size, entry.mode, entry.mtime = len(content), 0o644, 0
                    archive.addfile(entry, io.BytesIO(content))
            receipt = {"snapshot_id": snapshot_id, "filename": archive_path.name, "file_count": len(part),
                       "size": archive_path.stat().st_size, "sha256": file_hash(archive_path)}
            save_json(work / f"{name}.json", receipt)
            upload_files(api, args.repo_id, {f"{prefix}/{name}.tar": archive_path,
                         f"{prefix}/{name}.json": work / f"{name}.json"}, f"Upload dataset part {number}/{len(parts)}")
            receipts.append(receipt)
            archive_path.unlink()  # Only this temporary TAR, never source files.
        complete = {"status": "complete", "snapshot_id": snapshot_id,
                    "manifest_sha256": config.MANIFEST_SHA256, "source_project_root": str(project.project_root),
                    "selected_images_per_dataset": dict(Counter(r["split"] for r in rows)),
                    "file_count": len(records), "shards": receipts}
        save_json(work / "BACKUP_COMPLETE.json", complete)
        save_json(work / "LATEST_DATA_BACKUP.json", {"status": "complete", "snapshot_id": snapshot_id,
                  "path": prefix, "manifest_sha256": config.MANIFEST_SHA256})
        (work / "SHA256SUMS").write_text("".join(f"{r['sha256']}  {r['filename']}\n" for r in receipts)
                                       + f"{file_hash(work / 'FILE_INVENTORY.json')}  FILE_INVENTORY.json\n")
        commit = upload_files(api, args.repo_id, {f"{prefix}/BACKUP_COMPLETE.json": work / "BACKUP_COMPLETE.json",
                             f"{prefix}/SHA256SUMS": work / "SHA256SUMS",
                             "LATEST_DATA_BACKUP.json": work / "LATEST_DATA_BACKUP.json"}, "Complete prepared dataset backup")
    revision = commit.oid
    if re.fullmatch(r"[a-f0-9]{40}", revision) is None:
        raise ValueError("Hub did not return an immutable commit ID; inspect the completed remote backup.")
    print(f"DATA BACKUP COMPLETE: https://huggingface.co/datasets/{args.repo_id}/tree/{revision}/{prefix}", flush=True)
    print(f"Restore with --repo-id {args.repo_id} --snapshot {snapshot_id} --revision {revision}", flush=True)


if __name__ == "__main__":
    main()
