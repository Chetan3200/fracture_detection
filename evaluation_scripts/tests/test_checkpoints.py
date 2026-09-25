"""Synthetic CPU tests only: no models, network, credentials or GPU access."""
import hashlib
import importlib.util
import json
from pathlib import Path
import stat
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import warnings
import zipfile

SPEC = importlib.util.spec_from_file_location("checkpoints", Path(__file__).resolve().parents[1] / "checkpoints.py")
cp = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cp)
COMMIT = "a" * 40
BASE = "b" * 40


class ResolverTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.cache = self.root / "cache"
        self.run = self.root / "full_run"
        self.run.mkdir()
        manifest, prompt = b"synthetic split fixture\n", b"synthetic prompt\n"
        self.expected = hashlib.sha256(manifest).hexdigest()
        self.manifest_patch = patch.object(cp, "EXPECTED_MANIFEST_SHA256", self.expected)
        self.manifest_patch.start()
        self.addCleanup(self.manifest_patch.stop)
        self.addCleanup(self.tmp.cleanup)
        self.config = {"run_name": "full_run", "signature": {"mode": "full", "model_id": "example/base",
            "base_revision": BASE, "input_hashes": {"split_manifest.csv": self.expected,
            "prompt.txt": hashlib.sha256(prompt).hexdigest()}}}
        self.write("run_config.json", self.config)
        (self.run / "prompt.txt").write_bytes(prompt)
        (self.run / "frozen_split_manifest.csv").write_bytes(manifest)
        self.write("training_summary.json", {"mode": "full", "best_checkpoint": "/old/location/checkpoint-1680",
            "best_adapter": "/old/location/best_adapter", "selection_metric": "validation assistant-token loss, not localization AP"})
        self.write("best_adapter/adapter_config.json", {"base_model_name_or_path": "example/base", "revision": None})
        (self.run / "best_adapter/adapter_model.safetensors").write_bytes(b"NOT REAL MODEL WEIGHTS")
        for name in ("preprocessor_config.json", "tokenizer_config.json", "tokenizer.json"):
            self.write("processor/" + name, {})
        self.write("checkpoint-2520/optimizer.pt", {"never": "extract"})

    def write(self, name, value):
        dest = self.run / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(value), encoding="utf-8")

    def archive(self, name="backup.zip", extra=None):
        dest = self.root / name
        with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_STORED) as archive:
            for item in self.run.rglob("*"):
                if item.is_file():
                    archive.write(item, item.relative_to(self.run).as_posix())
            if extra:
                extra(archive)
        return dest

    def mock_hf(self, mapping, file_list=None):
        calls = []
        class API:
            def model_info(inner, **kwargs):
                calls.append(("info", kwargs))
                return types.SimpleNamespace(sha=COMMIT)
            def list_repo_files(inner, **kwargs):
                calls.append(("list", kwargs))
                return file_list if file_list is not None else list(mapping)
        def download(**kwargs):
            calls.append(("download", kwargs))
            return str(mapping[kwargs["filename"]])
        fake = types.ModuleType("huggingface_hub")
        fake.HfApi = API
        fake.hf_hub_download = download
        return patch.dict(sys.modules, huggingface_hub=fake), calls

    def mapping(self):
        archive = self.archive()
        marker = {"kind": "final", "run_name": "full_run", "archive_path": "runs/full_run/backups/final_step_002520_x.zip",
                  "archive_sha256": cp.sha256_file(archive), "checkpoint": "checkpoint-2520", "best_checkpoint": "checkpoint-1680"}
        pointer = self.root / "latest.json"
        pointer.write_text(json.dumps(marker))
        return {"runs/full_run/latest.json": pointer, "runs/full_run/run_config.json": self.run / "run_config.json",
                marker["archive_path"]: archive}, marker

    def test_import_is_standard_library_only(self):
        import ast
        tree = ast.parse(Path(cp.__file__).read_text())
        for node in tree.body:
            if isinstance(node, ast.ImportFrom):
                self.assertNotEqual(node.module, "huggingface_hub")
            if isinstance(node, ast.Import):
                self.assertNotIn("huggingface_hub", [alias.name for alias in node.names])

    def test_local_yolo_checksum(self):
        file = self.root / "weights.pt"
        file.write_bytes(b"synthetic")
        path, info = cp.resolve_yolo(file)
        self.assertEqual(path, file)
        self.assertEqual(info["weights_sha256"], cp.sha256_file(file))
        with self.assertRaises(ValueError):
            cp.resolve_yolo(file, repo="x/y")

    def test_hf_yolo_pinned_explicit_path(self):
        file = self.root / "extensionless-blob"
        file.write_bytes(b"synthetic")
        fake, calls = self.mock_hf({"nested/best.pt": file})
        with fake:
            _, info = cp.resolve_yolo(repo="x/y", filename="nested/best.pt", revision="tag", cache_dir=self.cache)
        self.assertEqual(info["resolved_revision"], COMMIT)
        self.assertEqual(calls[-1][1]["revision"], COMMIT)
        with self.assertRaises(ValueError):
            cp.resolve_yolo(repo="x/y")

    def test_local_full_and_adapter(self):
        for target in (self.run, self.run / "best_adapter"):
            run, adapter, info = cp.resolve_medgemma(target)
            self.assertEqual(run, self.run)
            self.assertEqual(adapter.name, "best_adapter")
            self.assertEqual(info["best_checkpoint"], "checkpoint-1680")
            self.assertEqual(info["base_revision"], BASE)

    def test_explicit_checkpoint(self):
        checkpoint = self.run / "checkpoint-2520"
        for name in ("adapter_config.json", "adapter_model.safetensors"):
            (checkpoint / name).write_bytes((self.run / "best_adapter" / name).read_bytes())
        _, adapter, info = cp.resolve_medgemma(checkpoint)
        self.assertEqual(adapter, checkpoint)
        self.assertEqual(info["selection"], "explicit_checkpoint")
        self.assertEqual(info["best_checkpoint"], "checkpoint-1680")

    def test_processor_in_best_adapter(self):
        for item in (self.run / "processor").iterdir():
            item.rename(self.run / "best_adapter" / item.name)
        (self.run / "processor").rmdir()
        _, _, info = cp.resolve_medgemma(self.run)
        self.assertEqual(info["processor_dir"], str(self.run / "best_adapter"))

    def test_frozen_manifest_and_prompt_checked(self):
        (self.run / "prompt.txt").write_text("changed")
        with self.assertRaisesRegex(ValueError, "prompt input hash"):
            cp.resolve_medgemma(self.run)
        (self.run / "frozen_split_manifest.csv").write_text("changed")
        with self.assertRaisesRegex(ValueError, "expected evaluation manifest"):
            cp.resolve_medgemma(self.run)

    def test_reject_unpinned_base(self):
        self.config["signature"]["base_revision"] = "main"
        self.write("run_config.json", self.config)
        with self.assertRaisesRegex(ValueError, "40-hex"):
            cp.resolve_medgemma(self.run)

    def test_selective_extraction_and_cached_hash_verification(self):
        archive = self.archive()
        run, _, info = cp.resolve_medgemma(archive, cache_dir=self.cache)
        self.assertFalse((run / "checkpoint-2520").exists())
        self.assertEqual(run.name, info["archive_sha256"])
        again, _, _ = cp.resolve_medgemma(archive, cache_dir=self.cache)
        self.assertEqual(again, run)
        weights = run / "best_adapter/adapter_model.safetensors"
        weights.write_bytes(b"X" * weights.stat().st_size)
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            cp.resolve_medgemma(archive, cache_dir=self.cache)

    def test_archive_checksum(self):
        with self.assertRaisesRegex(ValueError, "Archive SHA256"):
            cp._extract(self.archive(), self.cache, "0" * 64)

    def test_unsafe_ignored_zip_members(self):
        for index, name in enumerate(("../escape", "/abs", "checkpoint-3\\bad", "C:/evil", "a/./b")):
            archive = self.archive(f"unsafe{index}.zip", lambda z: z.writestr(name, "x"))
            with self.subTest(name=name), self.assertRaises(ValueError):
                cp._extract(archive, self.cache)

    def test_symlink_anywhere(self):
        def link(z):
            info = zipfile.ZipInfo("checkpoint-999/ignored")
            info.create_system = 3
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            z.writestr(info, "/etc/passwd")
        with self.assertRaisesRegex(ValueError, "Symlink"):
            cp._extract(self.archive(extra=link), self.cache)

    def test_duplicate_and_size_limits(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            archive = self.archive(extra=lambda z: z.writestr("prompt.txt", "duplicate"))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            cp._extract(archive, self.cache)
        with patch.object(cp, "MAX_SELECTED_BYTES", 1), self.assertRaisesRegex(ValueError, "safety limit"):
            cp._extract(self.archive("size.zip"), self.cache)

    def test_crc_and_no_partial_published_cache(self):
        archive = self.archive()
        archive.write_bytes(archive.read_bytes().replace(b"NOT REAL MODEL WEIGHTS", b"BAD REAL MODEL WEIGHTS"))
        with self.assertRaises(zipfile.BadZipFile):
            cp._extract(archive, self.cache)
        self.assertEqual(list((self.cache / "medgemma_extracted").iterdir()), [])

    def test_remote_pinning_and_best_not_final_checkpoint(self):
        mapping, _ = self.mapping()
        fake, calls = self.mock_hf(mapping)
        with fake:
            _, adapter, info = cp.resolve_medgemma(repo="x/y", revision="main", cache_dir=self.cache)
        self.assertEqual(adapter.name, "best_adapter")
        self.assertEqual(info["marker_checkpoint"], "checkpoint-2520")
        self.assertEqual(info["best_checkpoint"], "checkpoint-1680")
        for operation, args in calls:
            if operation != "info":
                self.assertEqual(args["revision"], COMMIT)

    def test_remote_smoke_and_unfinished_rejected(self):
        for field, value in (("kind", "checkpoint"), ("mode", "smoke")):
            mapping, marker = self.mapping()
            if field == "kind":
                marker[field] = value
                mapping["runs/full_run/latest.json"].write_text(json.dumps(marker))
            else:
                self.config["signature"][field] = value
                self.write("run_config.json", self.config)
            fake, _ = self.mock_hf(mapping)
            with fake, self.assertRaises(ValueError):
                cp.resolve_medgemma(repo="x/y", run="full_run", cache_dir=self.cache)

    def test_multiple_full_runs_require_explicit_selection(self):
        mapping, marker = self.mapping()
        marker2 = dict(marker, run_name="another", archive_path="runs/another/backups/final_step_002520_x.zip")
        config2 = dict(self.config, run_name="another")
        other_pointer, other_config = self.root / "other_latest.json", self.root / "other_config.json"
        other_pointer.write_text(json.dumps(marker2))
        other_config.write_text(json.dumps(config2))
        mapping.update({"runs/another/latest.json": other_pointer, "runs/another/run_config.json": other_config})
        fake, calls = self.mock_hf(mapping)
        with fake, self.assertRaisesRegex(ValueError, "another, full_run"):
            cp.resolve_medgemma(repo="x/y", cache_dir=self.cache)
        self.assertFalse(any(op == "download" and args["filename"].endswith(".zip") for op, args in calls))


if __name__ == "__main__":
    unittest.main()
