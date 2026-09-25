"""Restore a private sharded YOLO + MedGemma backup without overwriting data.

Python 3.9+. CLI > environment > .env > config.py. --help / --print-config are
offline. Use --revision <40-character commit> for exact reproduction; otherwise
resolve HEAD once, pin every download, and record that commit in the receipt.
Every archived file is verified against FILE_INVENTORY before rebasing data.yaml.
Historical source/cache paths in JSON remain provenance; MedGemma images are relative.
"""
import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
import tempfile

import config

DATA_FOLDERS = config.DATA_ARCHIVE_FOLDERS
file_hash = config.sha256


def check(condition, message):
    if not condition:
        raise ValueError(message)


def safe_relative(name):
    check(isinstance(name, str) and name and "\\" not in name and ":" not in name and "\x00" not in name,
          "Invalid inventory/archive path.")
    path = PurePosixPath(name)
    allowed = name == "split_manifest.csv" or any(name.startswith(f + "/") for f in DATA_FOLDERS)
    check(not path.is_absolute() and ".." not in path.parts and str(path) == name and allowed,
          f"Unexpected archive path: {name}")
    return path


def resolve_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", help="Private dataset repository; default DATASET_HF_REPO_ID")
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--manifest-path", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--snapshot", help="64-character completed snapshot ID; default latest at the pinned revision")
    parser.add_argument("--revision", help="Immutable 40-character HF commit ID; default resolve HEAD once")
    parser.add_argument("--print-config", action="store_true")
    args = parser.parse_args(argv)
    project = config.load_config(args.project_root, args.env_file)
    project = replace(project, **{name: config.resolve_path(getattr(args, name), project.project_root)
        for name in ("data_dir", "manifest_path", "cache_dir") if getattr(args, name) is not None})
    args.repo_id = args.repo_id.strip() if args.repo_id is not None else project.dataset_hf_repo_id
    if args.repo_id:
        check(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*", args.repo_id) is not None,
              "Dataset repository must be owner/repository, not a URL or credential.")
    check(args.repo_id or args.print_config, "Set DATASET_HF_REPO_ID in .env/environment or pass --repo-id.")
    if args.revision is not None:
        check(re.fullmatch(r"[a-f0-9]{40}", args.revision) is not None, "--revision must be a full immutable commit SHA.")
    if args.snapshot is not None:
        check(re.fullmatch(r"[a-f0-9]{64}", args.snapshot) is not None, "--snapshot must be a 64-character snapshot ID.")
    for destination in config.dataset_directories(project).values():
        check(not project.cache_dir.is_relative_to(destination), "Put CACHE_DIR outside the destination datasets.")
        check(not project.manifest_path.is_relative_to(destination), "Put MANIFEST_PATH outside the destination datasets.")
    return args, project


def protect_destinations(project):
    for destination in config.dataset_directories(project).values():
        if os.path.lexists(destination):
            raise FileExistsError(f"Dataset folder already exists: {destination}")
    if os.path.lexists(project.manifest_path):
        check(project.manifest_path.is_file() and file_hash(project.manifest_path) == config.MANIFEST_SHA256,
              "Existing project manifest differs from the frozen split.")


def validate_inventory(path, complete, snapshot, sums_path):
    inventory = json.loads(path.read_text(encoding="utf-8"))
    check(isinstance(inventory, dict), "Inventory must be a JSON object.")
    check(type(inventory.get("format_version")) is int and inventory["format_version"] in (1, 2),
          "Unsupported sharded inventory version.")
    check(inventory.get("manifest_sha256") == config.MANIFEST_SHA256, "Inventory uses a different split.")
    if inventory["format_version"] == 2:
        canonical = json.dumps(inventory, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        check(hashlib.sha256(canonical).hexdigest() == snapshot, "Inventory does not match its content-addressed snapshot ID.")
    checksums = {}
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        check(len(fields) == 2 and re.fullmatch(r"[a-f0-9]{64}", fields[0]) is not None, "Invalid SHA256SUMS line.")
        check(fields[1] not in checksums, "Duplicate SHA256SUMS entry.")
        checksums[fields[1]] = fields[0]
    check(checksums.get("FILE_INVENTORY.json") == file_hash(path), "FILE_INVENTORY checksum mismatch.")
    records = inventory.get("files")
    check(isinstance(records, list) and records and complete.get("file_count") == len(records), "Inventory file count mismatch.")
    expected = {}
    for record in records:
        check(isinstance(record, dict), "Invalid inventory record.")
        name = record.get("path")
        safe_relative(name)
        check(name not in expected, "Duplicate inventory path.")
        check(type(record.get("size")) is int and record["size"] >= 0, "Invalid inventory file size.")
        check(isinstance(record.get("sha256"), str) and re.fullmatch(r"[a-f0-9]{64}", record["sha256"]) is not None,
              "Invalid inventory file checksum.")
        expected[name] = record
    for name in expected:
        check(not any(str(parent) in expected for parent in PurePosixPath(name).parents),
              "Inventory has conflicting file and directory paths.")
    for name in ["split_manifest.csv"] + [f"{f}/split_manifest.csv" for f in DATA_FOLDERS]:
        check(name in expected and expected[name]["sha256"] == config.MANIFEST_SHA256, "Missing/different frozen manifest in inventory.")
    check(f"{DATA_FOLDERS[0]}/data.yaml" in expected, "YOLO data.yaml missing from inventory.")
    names, count = set(), 0
    for part in complete["shards"]:
        check(isinstance(part, dict), "Invalid shard receipt.")
        name = part.get("filename")
        check(isinstance(name, str) and re.fullmatch(r"part-\d{5}\.tar", name) is not None and name not in names,
              "Invalid or duplicate shard filename.")
        check(part.get("snapshot_id") == snapshot, "Shard receipt belongs to a different snapshot.")
        check(type(part.get("size")) is int and part["size"] > 0, "Invalid shard byte size.")
        check(type(part.get("file_count")) is int and part["file_count"] > 0, "Invalid shard file count.")
        check(checksums.get(name) == part.get("sha256") and name in checksums, "Shard checksum metadata mismatch.")
        names.add(name)
        count += part["file_count"]
    check(count == len(expected) and set(checksums) == names | {"FILE_INVENTORY.json"}, "Shard/inventory totals disagree.")
    return expected


def extract_part(archive_path, restored, expected, seen):
    """Never use extractall: accept only inventoried regular files, created exclusively."""
    count = 0
    with tarfile.open(archive_path, "r:") as archive:
        for member in archive:
            relative = safe_relative(member.name)
            check(member.isfile() and not member.issym() and not member.islnk() and member.sparse is None,
                  f"Non-regular archive entry: {member.name}")
            check(member.name in expected and member.name not in seen, f"Uninventoried/duplicate archive entry: {member.name}")
            check(member.size == expected[member.name]["size"], f"Inventory size mismatch: {member.name}")
            destination = restored.joinpath(*relative.parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            check(source is not None, f"Unreadable archive file: {member.name}")
            with source, destination.open("xb") as target:
                shutil.copyfileobj(source, target)
            seen.add(member.name)
            count += 1
    return count


def verify_restored(restored, expected, seen):
    check(seen == set(expected), "Restored file list does not exactly match inventory (missing/extra files).")
    for name, record in expected.items():
        file = restored.joinpath(*safe_relative(name).parts)
        check(file.is_file() and not file.is_symlink() and file.stat().st_size == record["size"]
              and file_hash(file) == record["sha256"], f"Restored file checksum/size mismatch: {name}")


def rebase_yaml(file, destination):
    """Change only the active top-level YAML path, after inventory verification."""
    before = file_hash(file)
    lines = file.read_bytes().decode("utf-8").splitlines(keepends=True)
    locations = [i for i, line in enumerate(lines) if re.match(r"^path\s*:", line)]
    check(len(locations) == 1, "Expected exactly one top-level path in YOLO data.yaml.")
    index = locations[0]
    ending = "\r\n" if lines[index].endswith("\r\n") else "\n" if lines[index].endswith("\n") else ""
    lines[index] = "path: " + json.dumps(str(destination)) + ending
    file.write_bytes("".join(lines).encode("utf-8"))
    return {"path": f"{DATA_FOLDERS[0]}/data.yaml", "before_sha256": before,
            "after_sha256": file_hash(file), "change": "Rebased active dataset path only"}


def publish_file(source, target):
    """Exclusive creation even if a competing process appears after preflight."""
    try:
        os.link(source, target)  # Fast on the same filesystem, never replaces a name.
    except FileExistsError:
        raise
    except OSError:
        # Cross-filesystem or a filesystem without hardlinks. xb still forbids overwrite.
        with source.open("rb") as reader, target.open("xb") as writer:
            shutil.copyfileobj(reader, writer)


def publish_datasets(restored, project):
    protect_destinations(project)
    destinations = config.dataset_directories(project)
    project.data_dir.mkdir(parents=True, exist_ok=True)
    # Reserve BOTH names exclusively before publishing any files. On an error,
    # keep partial new output visible for inspection; never delete existing data.
    for destination in destinations.values():
        destination.mkdir(exist_ok=False)
    for logical, destination in destinations.items():
        source = restored / logical
        for file in sorted(source.rglob("*")):
            if file.is_file():
                target = destination / file.relative_to(source)
                target.parent.mkdir(parents=True, exist_ok=True)
                publish_file(file, target)
    project.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if not os.path.lexists(project.manifest_path):
        try:
            publish_file(restored / "split_manifest.csv", project.manifest_path)
        except FileExistsError:
            check(file_hash(project.manifest_path) == config.MANIFEST_SHA256, "A different manifest appeared during restore.")
    else:
        check(file_hash(project.manifest_path) == config.MANIFEST_SHA256, "A different manifest appeared during restore.")


def main(argv=None):
    args, project = resolve_arguments(argv)
    if args.print_config:
        print(json.dumps({"action": "restore", "project": project.public_dict(), "repo_id": args.repo_id,
            "datasets": {k: str(v) for k, v in config.dataset_directories(project).items()}, "snapshot": args.snapshot,
            "revision": args.revision, "revision_policy": "Pin one immutable commit before downloading; record it in receipt",
            "manifest_sha256": config.MANIFEST_SHA256}, indent=2))
        return
    protect_destinations(project)
    project.activate_environment()
    from huggingface_hub import HfApi, hf_hub_download
    info = HfApi().repo_info(args.repo_id, repo_type="dataset", revision=args.revision)
    revision = info.sha
    check(info.private is True, "Expected a private Hugging Face dataset backup repository.")
    check(isinstance(revision, str) and re.fullmatch(r"[a-f0-9]{40}", revision) is not None, "Hub did not resolve an immutable commit.")
    check(args.revision is None or revision == args.revision, "Hub returned a different revision than requested.")
    print(f"Pinned HF revision: {revision}", flush=True)
    project.cache_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="dataset_download_", dir=project.cache_dir) as temporary:
        work = Path(temporary)
        def download(filename):
            return Path(hf_hub_download(repo_id=args.repo_id, repo_type="dataset", filename=filename,
                        revision=revision, local_dir=str(work / "downloads")))
        snapshot = args.snapshot
        if snapshot is None:
            latest = json.loads(download("LATEST_DATA_BACKUP.json").read_text())
            check(isinstance(latest, dict), "Latest pointer must be a JSON object.")
            check(latest.get("status") == "complete" and latest.get("manifest_sha256") == config.MANIFEST_SHA256,
                  "Latest pointer is incomplete or uses a different split.")
            snapshot = latest.get("snapshot_id")
        check(isinstance(snapshot, str) and re.fullmatch(r"[a-f0-9]{64}", snapshot) is not None, "Invalid snapshot ID.")
        prefix = f"snapshots/{snapshot}"
        complete = json.loads(download(f"{prefix}/BACKUP_COMPLETE.json").read_text())
        check(isinstance(complete, dict), "Completion record must be a JSON object.")
        check(complete.get("status") == "complete" and complete.get("snapshot_id") == snapshot
              and complete.get("manifest_sha256") == config.MANIFEST_SHA256
              and isinstance(complete.get("shards"), list) and complete["shards"], "Incomplete snapshot or different split.")
        inventory_path = download(f"{prefix}/FILE_INVENTORY.json")
        expected = validate_inventory(inventory_path, complete, snapshot, download(f"{prefix}/SHA256SUMS"))
        restored = work / "restored"
        restored.mkdir()
        seen = set()
        for number, part in enumerate(complete["shards"], 1):
            name = part["filename"]
            print(f"Downloading/verifying part {number}/{len(complete['shards'])}...", flush=True)
            archive = download(f"{prefix}/{name}")
            check(archive.stat().st_size == part["size"] and file_hash(archive) == part["sha256"], f"Archive checksum/size mismatch: {name}")
            check(extract_part(archive, restored, expected, seen) == part["file_count"], f"Shard file count mismatch: {name}")
            archive.unlink()  # Only this temporary downloaded TAR.
        verify_restored(restored, expected, seen)
        rewrite = rebase_yaml(restored / DATA_FOLDERS[0] / "data.yaml", project.yolo_data_dir)
        publish_datasets(restored, project)
        receipt = {"status": "complete", "repo_id": args.repo_id, "revision": revision, "snapshot_id": snapshot,
            "created_utc": datetime.now(timezone.utc).isoformat(), "manifest_sha256": config.MANIFEST_SHA256,
            "inventory_sha256": file_hash(inventory_path), "verified_file_count": len(expected),
            "verification": "Every archived file matched inventory SHA256 and size before the recorded YAML rebase",
            "datasets": {k: str(v) for k, v in config.dataset_directories(project).items()},
            "manifest_path": str(project.manifest_path), "post_verification_changes": [rewrite]}
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%fZ")
        receipt_path = project.data_dir / f"restore-{snapshot[:12]}-{stamp}.json"
        with receipt_path.open("x", encoding="utf-8") as handle:
            json.dump(receipt, handle, indent=2, allow_nan=False)
    print(f"DATA RESTORE COMPLETE: {project.data_dir}\nReceipt: {receipt_path}", flush=True)


if __name__ == "__main__":
    main()
