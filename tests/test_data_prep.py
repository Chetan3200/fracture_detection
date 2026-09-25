"""Offline stdlib checks for data-preparation entry points."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest

REPO = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DataPrepCliTests(unittest.TestCase):
    def run_script(self, script, *args, cwd=None):
        env = os.environ.copy()
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        return subprocess.run([PYTHON, "-B", str(script), *args], cwd=cwd,
                              env=env, text=True, capture_output=True, check=False)

    def test_help_from_unrelated_cwd_has_no_optional_imports(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in ("prepare_data.py", "prepare_medgemma.py"):
                result = self.run_script(REPO / "data_prep" / name, "--help", cwd=directory)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("--print-config", result.stdout)

    def test_print_config_precedence_and_root_anchoring(self):
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as elsewhere:
            root = Path(directory)
            (root / "custom.env").write_text("DATA_DIR=from_env\nCACHE_DIR=env_cache\nMANIFEST_PATH=env_manifest.csv\n")
            result = self.run_script(
                REPO / "data_prep" / "prepare_data.py", "--project-root", str(root),
                "--env-file", "custom.env", "--data-dir", "cli_data",
                "--manifest-path", "cli_manifest.csv", "--cache-dir", "cli_cache",
                "--print-config", cwd=elsewhere,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            values = json.loads(result.stdout)
            self.assertEqual(values["data_dir"], str((root / "cli_data").resolve()))
            self.assertEqual(values["manifest_path"], str((root / "cli_manifest.csv").resolve()))
            self.assertEqual(values["cache_dir"], str((root / "cli_cache").resolve()))
            self.assertEqual(values["env_file"], str((root / "custom.env").resolve()))

    def test_checksum_rejection_happens_before_kaggle_import(self):
        module = load_module("prepare_data_checksum", REPO / "data_prep" / "prepare_data.py")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "wrong.csv"
            manifest.write_text("not the frozen manifest\n")
            config = module.load_config(project_root=root, environ={"MANIFEST_PATH": str(manifest)})
            old_import = __import__("builtins").__import__
            calls = []
            def guarded_import(name, *args, **kwargs):
                calls.append(name)
                if name == "kagglehub":
                    raise AssertionError("Kaggle must not be imported for a bad manifest")
                return old_import(name, *args, **kwargs)
            __import__("builtins").__import__ = guarded_import
            try:
                with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                    module.build(config)
            finally:
                __import__("builtins").__import__ = old_import
            self.assertNotIn("kagglehub", calls)
            self.assertFalse((root / "data").exists())

    def test_existing_output_or_dangling_staging_is_protected(self):
        module = load_module("prepare_data_existing", REPO / "data_prep" / "prepare_data.py")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "split_manifest.csv"
            manifest.write_text("bad but unread because output protection is first\n")
            output = root / "data" / "grazpedwri_yolo"
            output.parent.mkdir()
            output.symlink_to(root / "missing-target")
            config = module.load_config(project_root=root)
            with self.assertRaises(FileExistsError):
                module.build(config)


class MedGemmaConversionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_module("prepare_medgemma_conversion", REPO / "data_prep" / "prepare_medgemma.py")

    def test_box_conversion_and_targets_are_preserved(self):
        converted, clamps, error = self.module.convert_boxes([[0, 0.5, 0.5, 0.2, 0.4]], "sample")
        self.assertEqual(converted, [{"box_2d": [300.0, 400.0, 700.0, 600.0], "label": "fracture"}])
        self.assertEqual(clamps, 0)
        self.assertLessEqual(error, 0.000005)
        self.assertEqual(self.module.target_text(converted), 'Final Answer: ```json\n[{"box_2d":[300.0,400.0,700.0,600.0],"label":"fracture"}]\n```')
        record = self.module.model_record("id", "1", "val", "images/val/id.png")
        self.assertEqual(len(record["messages"]), 1)
        self.assertEqual(record["messages"][0]["content"][1]["text"], self.module.PROMPT)


class MedGemmaConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_module(
            "prepare_medgemma_config",
            REPO / "data_prep" / "prepare_medgemma.py",
        )
        cls.frozen_manifest = (REPO / "split_manifest.csv").read_bytes()

    def write_prepared_source(self, root, manifest_bytes=None):
        manifest_bytes = (
            self.frozen_manifest if manifest_bytes is None else manifest_bytes
        )
        source = root / "data" / "grazpedwri_yolo"
        source.mkdir(parents=True)
        manifest = source / "split_manifest.csv"
        manifest.write_bytes(manifest_bytes)
        (source / "dataset_info.json").write_text(json.dumps({
            "dataset": self.module.DATASET,
            "manifest_sha256": self.module.sha256(manifest),
            "dataset_root": str(root / "unused-source-images"),
        }))

    def test_build_requires_configured_root_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_prepared_source(root)
            config = self.module.load_config(
                project_root=root,
                environ={"MANIFEST_PATH": "configured/missing.csv"},
            )
            with self.assertRaisesRegex(
                FileNotFoundError,
                "Configured root manifest",
            ):
                self.module.build(config)

    def test_build_rejects_wrong_configured_root_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_prepared_source(root)
            (root / "split_manifest.csv").write_text("wrong root manifest\n")
            config = self.module.load_config(project_root=root)
            with self.assertRaisesRegex(ValueError, "Configured root manifest SHA256 mismatch"):
                self.module.build(config)

    def test_build_rejects_wrong_prepared_source_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "split_manifest.csv").write_bytes(self.frozen_manifest)
            self.write_prepared_source(root, b"wrong prepared source manifest\n")
            config = self.module.load_config(project_root=root)
            with self.assertRaisesRegex(ValueError, "Frozen manifest SHA256 mismatch"):
                self.module.build(config)

    def test_build_protects_dangling_output_and_staging(self):
        for name in ("grazpedwri_medgemma", "grazpedwri_medgemma.building"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                location = root / "data" / name
                location.parent.mkdir()
                location.symlink_to(root / "missing-target")
                config = self.module.load_config(project_root=root)
                with self.assertRaises(FileExistsError):
                    self.module.build(config)

    def test_dataset_helpers_use_configured_and_relative_root_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.module.load_config(
                project_root=root,
                environ={"DATA_DIR": "configured-data"},
            )
            calls = []

            class FakeDataset:
                def map(self, function):
                    self.resolved = function({"image": "images/val/example.png"})
                    return self

                def cast_column(self, name, image_type):
                    self.cast = (name, image_type)
                    return self

            class FakeDatasetDict(dict):
                def map(self, function):
                    for dataset in self.values():
                        dataset.map(function)
                    return self

                def cast_column(self, name, image_type):
                    for dataset in self.values():
                        dataset.cast_column(name, image_type)
                    return self

            def fake_load_dataset(kind, data_files):
                calls.append((kind, data_files))
                return FakeDatasetDict(
                    train=FakeDataset(),
                    validation=FakeDataset(),
                    val=FakeDataset(),
                    test=FakeDataset(),
                )

            fake_datasets = types.ModuleType("datasets")
            fake_datasets.Image = type("FakeImage", (), {})
            fake_datasets.load_dataset = fake_load_dataset
            old_datasets = sys.modules.get("datasets")
            old_load_config = self.module.load_config
            self.module.load_config = lambda: config
            sys.modules["datasets"] = fake_datasets
            try:
                default_root = config.medgemma_data_dir
                (default_root / "images" / "val").mkdir(parents=True)
                (default_root / "images" / "val" / "example.png").write_bytes(b"x")
                self.module.load_sft_datasets()
                self.assertEqual(
                    calls[0][1]["train"],
                    str(default_root / "train.jsonl"),
                )

                relative_root = root / "relative-medgemma"
                (relative_root / "images" / "val").mkdir(parents=True)
                (relative_root / "images" / "val" / "example.png").write_bytes(b"x")
                self.module.load_inference_dataset("val", "relative-medgemma")
                self.assertEqual(
                    calls[1][1]["val"],
                    str(relative_root / "val_inputs.jsonl"),
                )
            finally:
                self.module.load_config = old_load_config
                if old_datasets is None:
                    del sys.modules["datasets"]
                else:
                    sys.modules["datasets"] = old_datasets


if __name__ == "__main__":
    unittest.main()
