"""Offline, synthetic coverage for the sharded dataset transfer entry points."""
import contextlib
import errno
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


backup = load_module("backup_data_transfer_under_test", "backup_data_to_hf.py")
restore = load_module("restore_data_transfer_under_test", "download_data_from_hf.py")
config = backup.config


class Add:
    def __init__(self, path_in_repo, path_or_fileobj):
        self.path_in_repo = path_in_repo
        self.path_or_fileobj = path_or_fileobj


class FakeHub:
    """An in-memory Hub: commits retain bytes; downloads only write caller temp files."""
    oid = "a" * 40

    def __init__(self, private=True, fail_commit=None, exists=False):
        self.private, self.fail_commit, self.exists = private, fail_commit, exists
        self.blobs, self.calls, self.downloads = {}, [], []
        self.commit_count = 0

    def create_repo(self, **kwargs):
        self.calls.append(("create_repo", kwargs))

    def repo_info(self, *args, **kwargs):
        self.calls.append(("repo_info", {"args": args, **kwargs}))
        return SimpleNamespace(private=self.private, sha=self.oid)

    def file_exists(self, *args, **kwargs):
        self.calls.append(("file_exists", {"args": args, **kwargs}))
        return self.exists

    def create_commit(self, **kwargs):
        self.calls.append(("create_commit", kwargs))
        self.commit_count += 1
        if self.fail_commit == self.commit_count:
            raise OSError("synthetic upload failure")
        for operation in kwargs["operations"]:
            value = operation.path_or_fileobj
            self.blobs[operation.path_in_repo] = value if isinstance(value, bytes) else Path(value).read_bytes()
        return SimpleNamespace(oid=self.oid)

    def download(self, **kwargs):
        self.downloads.append(kwargs)
        target = Path(kwargs["local_dir"]) / kwargs["filename"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(self.blobs[kwargs["filename"]])
        return str(target)

    @contextlib.contextmanager
    def installed(self):
        sdk = types.ModuleType("huggingface_hub")
        sdk.CommitOperationAdd = Add
        sdk.HfApi = lambda: self
        sdk.hf_hub_download = self.download
        with patch.dict(sys.modules, {"huggingface_hub": sdk}):
            yield self


def sha(data):
    return hashlib.sha256(data).hexdigest()


def tar_bytes(entries):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        for name, content, kind in entries:
            info = tarfile.TarInfo(name)
            if kind == "file":
                info.size, info.mode, info.mtime = len(content), 0o644, 0
                archive.addfile(info, io.BytesIO(content))
            elif kind == "symlink":
                info.type, info.linkname = tarfile.SYMTYPE, "elsewhere"
                archive.addfile(info)
            elif kind == "hardlink":
                info.type, info.linkname = tarfile.LNKTYPE, "elsewhere"
                archive.addfile(info)
    return stream.getvalue()


class TransferTests(unittest.TestCase):
    def make_source(self, base):
        """Tiny frozen data with source paths deliberately outside the project root."""
        base = Path(base)
        base.mkdir(parents=True, exist_ok=True)
        data, cache = base / "outside-source-data", base / "cache"
        manifest = base / "outside-source-manifest.csv"
        text = ("sample_type,image_relpath,split,filestem\n"
                "positive,originals/case.jpg,train,case\n"
                "excluded,originals/skip.jpg,test,skip\n")
        manifest.write_text(text, encoding="utf-8")
        yolo, med = data / "grazpedwri_yolo", data / "grazpedwri_medgemma"
        for folder in (yolo, med):
            folder.mkdir(parents=True)
            (folder / "split_manifest.csv").write_text(text, encoding="utf-8")
        external_image = base / "actual-image.jpg"
        external_image.write_bytes(b"\x89PNG\r\nsynthetic-yolo-image")
        (yolo / "images/train").mkdir(parents=True)
        (yolo / "images/train/case.jpg").symlink_to(external_image)
        (yolo / "labels/train").mkdir(parents=True)
        (yolo / "labels/train/case.txt").write_text("0 0.5 0.5 1 1\n")
        yaml = b'path: "/historical/source/yolo"\ntrain: images/train\nval: images/val\n'
        (yolo / "data.yaml").write_bytes(yaml)
        (med / "images/train").mkdir(parents=True)
        med_image = b"\x89PNG\r\nsynthetic-medgemma-image"
        (med / "images/train/case.png").write_bytes(med_image)
        provenance = {"image": "images/train/case.png", "source_path": "/historical/cache/case.png"}
        (med / "records.json").write_text(json.dumps(provenance, sort_keys=True))
        digest = sha(text.encode())
        return data, manifest, cache, digest, external_image.read_bytes(), med_image, provenance, yaml

    def backup_args(self, root, data, manifest, cache, extra=()):
        return ["--project-root", str(root), "--data-dir", str(data), "--manifest-path", str(manifest),
                "--cache-dir", str(cache), "--repo-id", "test/private", "--shard-mib", "1", *extra]

    def do_backup(self, root, data, manifest, cache, digest, api=None):
        api = api or FakeHub()
        with patch.object(config, "MANIFEST_SHA256", digest), api.installed():
            backup.main(self.backup_args(root, data, manifest, cache))
        return api

    def snapshot(self, api):
        latest = json.loads(api.blobs["LATEST_DATA_BACKUP.json"])
        return latest["snapshot_id"], latest["path"]

    def test_full_backup_restore_preserves_bytes_provenance_and_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data, manifest, cache, digest, yolo_bytes, med_bytes, provenance, original_yaml = self.make_source(root)
            api = self.do_backup(root, data, manifest, cache, digest)
            snapshot, prefix = self.snapshot(api)
            self.assertRegex(snapshot, r"^[a-f0-9]{64}$")
            # Tar contains dereferenced regular image bytes, not the source symlink.
            part_name = json.loads(api.blobs[prefix + "/BACKUP_COMPLETE.json"])["shards"][0]["filename"]
            with tarfile.open(fileobj=io.BytesIO(api.blobs[prefix + "/" + part_name])) as archive:
                image = archive.getmember("data/grazpedwri_yolo/images/train/case.jpg")
                self.assertTrue(image.isfile())
                self.assertFalse(image.issym())
                self.assertEqual(archive.extractfile(image).read(), yolo_bytes)

            target_data, target_manifest, target_cache = root / "restored-data", root / "restored.csv", root / "restore-cache"
            args = ["--project-root", str(root), "--data-dir", str(target_data), "--manifest-path", str(target_manifest),
                    "--cache-dir", str(target_cache), "--repo-id", "test/private", "--snapshot", snapshot]
            with patch.object(config, "MANIFEST_SHA256", digest), api.installed():
                restore.main(args)
            self.assertEqual((target_data / "grazpedwri_yolo/images/train/case.jpg").read_bytes(), yolo_bytes)
            self.assertEqual((target_data / "grazpedwri_medgemma/images/train/case.png").read_bytes(), med_bytes)
            self.assertEqual(json.loads((target_data / "grazpedwri_medgemma/records.json").read_text()), provenance)
            rebased = (target_data / "grazpedwri_yolo/data.yaml").read_text()
            self.assertIn(json.dumps(str(target_data / "grazpedwri_yolo")), rebased)
            self.assertNotEqual(rebased.encode(), original_yaml)  # The documented sole post-verify change.
            receipt = json.loads(next(target_data.glob("restore-*.json")).read_text())
            self.assertEqual(receipt["revision"], api.oid)
            self.assertEqual(receipt["snapshot_id"], snapshot)
            self.assertEqual(receipt["manifest_sha256"], digest)
            self.assertEqual(receipt["verified_file_count"], len(json.loads(api.blobs[prefix + "/FILE_INVENTORY.json"])["files"]))
            self.assertEqual(receipt["post_verification_changes"][0]["before_sha256"], sha(original_yaml))
            self.assertEqual(receipt["post_verification_changes"][0]["after_sha256"], sha(rebased.encode()))
            self.assertTrue(api.downloads)
            self.assertTrue(all(call["revision"] == api.oid for call in api.downloads))

    def test_restore_destination_and_manifest_protection_precedes_sdk(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data, manifest, cache, digest, *_ = self.make_source(root)
            project = config.load_config(root, environ={"DATA_DIR": str(data), "MANIFEST_PATH": str(manifest), "CACHE_DIR": str(cache)})
            api = FakeHub()
            with patch.object(config, "MANIFEST_SHA256", digest), api.installed():
                with self.assertRaises(FileExistsError):
                    restore.main(["--project-root", str(root), "--data-dir", str(data), "--manifest-path", str(manifest),
                                  "--cache-dir", str(cache), "--repo-id", "test/private"])
            self.assertEqual(api.calls, [])
            # A dangling destination is also an existing destination, before any API call.
            data2, manifest2, cache2, digest2, *_ = self.make_source(root / "second")
            dangling = data2 / "grazpedwri_yolo"
            for child in dangling.iterdir():
                if child.is_dir():
                    import shutil; shutil.rmtree(child)
                else: child.unlink()
            dangling.rmdir(); dangling.symlink_to(root / "missing-target")
            api2 = FakeHub()
            with patch.object(config, "MANIFEST_SHA256", digest2), api2.installed():
                with self.assertRaises(FileExistsError):
                    restore.main(["--project-root", str(root / "second"), "--data-dir", str(data2),
                                  "--manifest-path", str(manifest2), "--cache-dir", str(cache2),
                                  "--repo-id", "test/private"])
            self.assertEqual(api2.calls, [])
            bad_manifest = root / "bad-root-manifest.csv"; bad_manifest.write_text("not frozen")
            clean_data = root / "clean-data"
            api3 = FakeHub()
            with patch.object(config, "MANIFEST_SHA256", digest), api3.installed():
                with self.assertRaisesRegex(ValueError, "Existing project manifest"):
                    restore.main(["--project-root", str(root), "--data-dir", str(clean_data), "--manifest-path", str(bad_manifest),
                                  "--cache-dir", str(cache), "--repo-id", "test/private"])
            self.assertEqual(api3.calls, [])

    def test_archive_and_content_tampering_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); data, manifest, cache, digest, *_ = self.make_source(root)
            api = self.do_backup(root, data, manifest, cache, digest); snapshot, prefix = self.snapshot(api)
            complete = json.loads(api.blobs[prefix + "/BACKUP_COMPLETE.json"]); part = complete["shards"][0]
            # Corruption is detected from receipt hash before extraction.
            api.blobs[prefix + "/" + part["filename"]] = b"not a tar"
            target = root / "bad1"
            with patch.object(config, "MANIFEST_SHA256", digest), api.installed(), self.assertRaisesRegex(ValueError, "Archive checksum"):
                restore.main(["--project-root", str(root), "--data-dir", str(target), "--manifest-path", str(root / "bad1.csv"), "--cache-dir", str(root / "c1"), "--repo-id", "test/private", "--snapshot", snapshot])

            # Rebuild a valid TAR whose image byte differs, then honestly update TAR receipt/sums.
            api = self.do_backup(root, data, manifest, cache, digest, FakeHub()); snapshot, prefix = self.snapshot(api)
            complete = json.loads(api.blobs[prefix + "/BACKUP_COMPLETE.json"]); part = complete["shards"][0]
            old = api.blobs[prefix + "/" + part["filename"]]
            altered = io.BytesIO()
            with tarfile.open(fileobj=io.BytesIO(old), mode="r:") as src, tarfile.open(fileobj=altered, mode="w") as dst:
                for member in src:
                    content = src.extractfile(member).read()
                    if member.name.endswith("case.jpg"): content = b"T" * len(content)
                    entry = tarfile.TarInfo(member.name); entry.size = len(content); entry.mode = 0o644
                    dst.addfile(entry, io.BytesIO(content))
            changed = altered.getvalue(); api.blobs[prefix + "/" + part["filename"]] = changed
            part["size"], part["sha256"] = len(changed), sha(changed)
            api.blobs[prefix + "/BACKUP_COMPLETE.json"] = json.dumps(complete, sort_keys=True, separators=(",", ":")).encode()
            sums_name = prefix + "/SHA256SUMS"; lines = api.blobs[sums_name].decode().splitlines()
            api.blobs[sums_name] = ("\n".join((part["sha256"] + "  " + part["filename"]) if line.endswith("  " + part["filename"]) else line for line in lines) + "\n").encode()
            bad_data, bad_manifest = root / "bad2", root / "bad2.csv"
            with patch.object(config, "MANIFEST_SHA256", digest), api.installed(), self.assertRaisesRegex(ValueError, "checksum/size"):
                restore.main(["--project-root", str(root), "--data-dir", str(bad_data), "--manifest-path", str(bad_manifest), "--cache-dir", str(root / "c2"), "--repo-id", "test/private", "--snapshot", snapshot])
            # Integrity failures never publish either newly restored dataset or a manifest.
            self.assertFalse((bad_data / "grazpedwri_yolo").exists())
            self.assertFalse((bad_data / "grazpedwri_medgemma").exists())
            self.assertFalse(bad_manifest.exists())

    def test_extract_rejects_traversal_links_duplicates_unlisted_and_missing(self):
        expected = {"split_manifest.csv": {"size": 1, "sha256": sha(b"x")}}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); restored = root / "restored"; restored.mkdir()
            for name, kind in [("../escape", "file"), ("split_manifest.csv", "symlink"), ("split_manifest.csv", "hardlink"), ("other.txt", "file")]:
                archive = root / (kind + ".tar"); archive.write_bytes(tar_bytes([(name, b"x", kind)]))
                with self.assertRaises(ValueError): restore.extract_part(archive, restored, expected, set())
            duplicate = root / "duplicate.tar"; duplicate.write_bytes(tar_bytes([( "split_manifest.csv", b"x", "file"), ("split_manifest.csv", b"x", "file")]))
            with self.assertRaises(ValueError): restore.extract_part(duplicate, restored, expected, set())
            empty = root / "empty.tar"; empty.write_bytes(tar_bytes([]))
            self.assertEqual(restore.extract_part(empty, root / "empty-out", expected, set()), 0)
            with self.assertRaisesRegex(ValueError, "exactly match"):
                restore.verify_restored(root / "empty-out", expected, set())
            for bad in ["/absolute", "a\\b", "a/../b", "data/grazpedwri_yolo/../../x", "x:bad"]:
                with self.assertRaises(ValueError): restore.safe_relative(bad)

    def test_inventory_validation_rejects_malformed_receipts_and_totals(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); manifest = b"frozen"; digest = sha(manifest)
            inventory = {"format_version": 2, "manifest_sha256": digest, "files": [
                {"path": "split_manifest.csv", "size": len(manifest), "sha256": digest},
                {"path": "data/grazpedwri_yolo/split_manifest.csv", "size": len(manifest), "sha256": digest},
                {"path": "data/grazpedwri_medgemma/split_manifest.csv", "size": len(manifest), "sha256": digest},
                {"path": "data/grazpedwri_yolo/data.yaml", "size": 1, "sha256": sha(b"x")} ]}
            snapshot = sha(json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode())
            inv = root / "inventory"; inv.write_text(json.dumps(inventory, sort_keys=True, separators=(",", ":")))
            sums = root / "sums"; sums.write_text(sha(inv.read_bytes()) + "  FILE_INVENTORY.json\n" + "b" * 64 + "  part-00001.tar\n")
            complete = {"file_count": 4, "shards": [{"filename": "part-00001.tar", "snapshot_id": snapshot, "size": 1, "file_count": 4, "sha256": "b" * 64}]}
            with patch.object(config, "MANIFEST_SHA256", digest):
                restore.validate_inventory(inv, complete, snapshot, sums)
                for altered in [
                    {**complete, "file_count": 3},
                    {**complete, "shards": [{**complete["shards"][0], "file_count": 3}]},
                    {**complete, "shards": [{**complete["shards"][0], "filename": "oops.tar"}]},
                ]:
                    with self.assertRaises(ValueError): restore.validate_inventory(inv, altered, snapshot, sums)
                bad_inventory = dict(inventory); bad_inventory["files"] = inventory["files"] + [inventory["files"][0]]
                inv.write_text(json.dumps(bad_inventory));
                with self.assertRaises(ValueError): restore.validate_inventory(inv, complete, snapshot, sums)

    def test_upload_privacy_failure_and_canonical_layout(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); data, manifest, cache, digest, *_ = self.make_source(root)
            public = FakeHub(private=False)
            with patch.object(config, "MANIFEST_SHA256", digest), public.installed(), self.assertRaisesRegex(ValueError, "private"):
                backup.main(self.backup_args(root, data, manifest, cache))
            self.assertFalse(any(call[0] == "create_commit" for call in public.calls))
            failing = FakeHub(fail_commit=3)
            with patch.object(config, "MANIFEST_SHA256", digest), failing.installed(), self.assertRaises(OSError):
                backup.main(self.backup_args(root, data, manifest, cache))
            self.assertFalse(any(name.endswith("BACKUP_COMPLETE.json") for name in failing.blobs))
            good = self.do_backup(root, data, manifest, cache, digest, FakeHub())
            snapshot, prefix = self.snapshot(good)
            self.assertEqual(prefix, "snapshots/" + snapshot)
            inventory = json.loads(good.blobs[prefix + "/FILE_INVENTORY.json"])
            self.assertTrue(all(p["path"] == "split_manifest.csv" or p["path"].startswith(config.DATA_ARCHIVE_FOLDERS) for p in inventory["files"]))
            self.assertTrue(all(r["filename"].startswith("part-") and r["filename"].endswith(".tar") for r in json.loads(good.blobs[prefix + "/BACKUP_COMPLETE.json"])["shards"]))

    def test_cli_env_precedence_and_offline_help_print_config(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); (root / ".env").write_text("DATASET_HF_REPO_ID=dot/env\n")
            with patch.dict(os.environ, {"DATASET_HF_REPO_ID": "process/env"}, clear=False):
                self.assertEqual(backup.resolve_arguments(["--project-root", str(root)])[0].repo_id, "process/env")
                self.assertEqual(restore.resolve_arguments(["--project-root", str(root), "--repo-id", "cli/repo"])[0].repo_id, "cli/repo")
            old_env = os.environ.copy(); old_env["PYTHONPATH"] = str(root) + os.pathsep + str(ROOT)
            # A poison SDK proves these parser-only paths never import it, from an unrelated CWD.
            (root / "huggingface_hub.py").write_text("raise RuntimeError('SDK must not import')\n")
            for script in ("backup_data_to_hf.py", "download_data_from_hf.py"):
                help_run = subprocess.run([sys.executable, "-B", str(ROOT / script), "--help"], cwd=str(root), env=old_env, capture_output=True, text=True)
                self.assertEqual(help_run.returncode, 0, help_run.stderr)
                printed = subprocess.run([sys.executable, "-B", str(ROOT / script), "--project-root", str(root), "--print-config"], cwd=str(root), env=old_env, capture_output=True, text=True)
                self.assertEqual(printed.returncode, 0, printed.stderr)
                self.assertIn('"repo_id": "dot/env"', printed.stdout)

    def test_two_shard_latest_restore_preserves_all_source_bytes_and_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data, manifest, cache, digest, *_ = self.make_source(root)
            # Two individual image bodies below 1 MiB sum to more than one shard.
            yolo_body = b"Y" * (600 * 1024)
            med_body = b"M" * (600 * 1024)
            (root / "actual-image.jpg").write_bytes(yolo_body)  # Existing image is a source symlink.
            (data / "grazpedwri_medgemma/images/train/case.png").write_bytes(med_body)
            api = self.do_backup(root, data, manifest, cache, digest)
            snapshot, prefix = self.snapshot(api)
            complete = json.loads(api.blobs[prefix + "/BACKUP_COMPLETE.json"])
            self.assertGreaterEqual(len(complete["shards"]), 2)
            self.assertEqual(sum(shard["file_count"] for shard in complete["shards"]), complete["file_count"])

            target_data, target_manifest, target_cache = root / "large-restore", root / "existing-frozen.csv", root / "large-cache"
            target_manifest.write_bytes(manifest.read_bytes())  # An identical frozen root manifest is protected/preserved.
            args = ["--project-root", str(root), "--data-dir", str(target_data), "--manifest-path", str(target_manifest),
                    "--cache-dir", str(target_cache), "--repo-id", "test/private", "--revision", api.oid]
            with patch.object(config, "MANIFEST_SHA256", digest), api.installed():
                restore.main(args)  # No --snapshot: exercise LATEST_DATA_BACKUP.json at the explicit revision.
            self.assertEqual(api.downloads[0]["filename"], "LATEST_DATA_BACKUP.json")
            self.assertTrue(all(call["revision"] == api.oid for call in api.downloads))
            self.assertEqual(target_manifest.read_bytes(), manifest.read_bytes())
            for source_root, destination_root in [
                (data / "grazpedwri_yolo", target_data / "grazpedwri_yolo"),
                (data / "grazpedwri_medgemma", target_data / "grazpedwri_medgemma"),
            ]:
                for source in source_root.rglob("*"):
                    if source.is_file() and source.relative_to(source_root).as_posix() != "data.yaml":
                        self.assertEqual((destination_root / source.relative_to(source_root)).read_bytes(), source.read_bytes())
            restored_yaml = target_data / "grazpedwri_yolo/data.yaml"
            self.assertNotEqual(restored_yaml.read_bytes(), (data / "grazpedwri_yolo/data.yaml").read_bytes())
            receipt = json.loads(next(target_data.glob("restore-*.json")).read_text())
            self.assertEqual(receipt["inventory_sha256"], sha(api.blobs[prefix + "/FILE_INVENTORY.json"]))
            self.assertEqual(receipt["verified_file_count"], complete["file_count"])
            self.assertEqual(receipt["revision"], api.oid)

    def test_restore_rejects_nonimmutable_or_mismatched_revision_before_download(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for revision in ["main", "a" * 39, "A" * 40, "a" * 40 + "0"]:
                with self.assertRaisesRegex(ValueError, "revision"):
                    restore.resolve_arguments(["--project-root", str(root), "--repo-id", "test/private", "--revision", revision])
            data, manifest, cache, digest, *_ = self.make_source(root)
            api = self.do_backup(root, data, manifest, cache, digest)
            requested, api.oid = "a" * 40, "b" * 40
            with patch.object(config, "MANIFEST_SHA256", digest), api.installed(), self.assertRaisesRegex(ValueError, "different revision"):
                restore.main(["--project-root", str(root), "--data-dir", str(root / "mismatch-data"),
                              "--manifest-path", str(root / "mismatch.csv"), "--cache-dir", str(root / "mismatch-cache"),
                              "--repo-id", "test/private", "--revision", requested])
            self.assertEqual(api.downloads, [])

    def test_publish_file_exclusive_and_exdev_fallback(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, target = root / "source", root / "target"
            source.write_bytes(b"exact synthetic bytes")
            target.write_bytes(b"existing bytes")
            with self.assertRaises(FileExistsError):
                restore.publish_file(source, target)
            self.assertEqual(target.read_bytes(), b"existing bytes")
            fallback = root / "fallback"
            with patch.object(restore.os, "link", side_effect=OSError(errno.EXDEV, "cross-device")):
                restore.publish_file(source, fallback)
            self.assertEqual(fallback.read_bytes(), source.read_bytes())

    def test_backup_manifest_and_restore_cache_guards_precede_hub(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data, manifest, cache, digest, *_ = self.make_source(root)
            api = FakeHub()
            with patch.object(config, "MANIFEST_SHA256", "0" * 64), api.installed(), self.assertRaisesRegex(ValueError, "Frozen manifest checksum"):
                backup.main(self.backup_args(root, data, manifest, cache))
            self.assertEqual(api.calls, [])
            with self.assertRaisesRegex(ValueError, "CACHE_DIR outside"):
                restore.resolve_arguments(["--project-root", str(root), "--repo-id", "test/private", "--data-dir", str(data),
                                           "--manifest-path", str(manifest), "--cache-dir", str(data / "grazpedwri_yolo/cache")])


if __name__ == "__main__":
    unittest.main()
