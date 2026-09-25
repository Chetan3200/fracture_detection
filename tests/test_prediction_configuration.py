"""Offline configuration contracts for prediction_cli.

These tests use temporary paths and synthetic protocol reports only.  They never
load a model, contact a remote service, or invoke a real prediction backend.
"""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import config
import prediction_cli as cli


class PredictionConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.env = patch.dict(os.environ, {}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)

    def write_env(self, text=""):
        (self.root / ".env").write_text(text, encoding="utf-8")

    def resolve(self, mode, *args):
        return cli.resolve_arguments(mode, ["--project-root", str(self.root), *args])

    def plan(self, mode, *args):
        parsed, project = self.resolve(mode, *args)
        return parsed, project, cli.make_plan(parsed, project)

    def protocol(self, mode="evaluate", model="yolo26", **changes):
        files = cli.source_files(mode, model)
        checked = files if mode == "evaluate" else files[2:]
        inference = ({"collection": {"imgsz": 960}, "prediction_batch": 4}
                     if model == "yolo26" else {"max_new_tokens": 777})
        identity = {
            "model": model,
            "raw_manifest_sha256": config.MANIFEST_SHA256,
            "source_sha256": {item.name: config.sha256(item) for item in checked},
            "score_floor": 0.001,
            "max_detections": 300,
            "confidence_comparison": "score > cutoff",
            "inference": inference,
        }
        report = {"status": "complete", "split": "val", "protocol": {
            "selected_on": "val", "confidence_cutoff": 0.25, "identity": identity}}
        for dotted, value in changes.items():
            target = report
            parts = dotted.split("__")
            for key in parts[:-1]:
                target = target[key]
            target[parts[-1]] = value
        path = self.root / "complete_validation.json"
        path.write_text(json.dumps(report), encoding="utf-8")
        return path, report

    def test_paths_are_root_relative_not_cwd(self):
        self.write_env("DATA_DIR=dotenv-data\nRUNS_DIR=dotenv-runs\nCACHE_DIR=dotenv-cache\n")
        elsewhere = self.root / "elsewhere"
        elsewhere.mkdir()
        old = Path.cwd()
        os.chdir(elsewhere)
        try:
            args, project = self.resolve("evaluate", "yolo26", "--checkpoint", "weights/a.pt")
        finally:
            os.chdir(old)
        self.assertEqual(project.data_dir, self.root / "dotenv-data")
        self.assertEqual(project.runs_dir, self.root / "dotenv-runs")
        self.assertEqual(args.checkpoint, self.root / "weights/a.pt")
        self.assertEqual(args.cache_dir, self.root / "dotenv-cache/fracture_evaluation")

    def test_process_env_beats_dotenv_and_cli_beats_both(self):
        self.write_env("RUNS_DIR=dotenv-runs\nHF_REPO_ID=dotenv/yolo\nPREDICTION_DEVICE=2\nPREDICTION_PYTHON=dotenv-python\n")
        os.environ.update(RUNS_DIR="process-runs", HF_REPO_ID="process/yolo", PREDICTION_DEVICE="3",
                          PREDICTION_PYTHON="process-python")
        args, project = self.resolve("evaluate", "yolo26", "--hf", "--hf-repo", "cli/yolo",
                                     "--device", "4", "--python", "cli-python", "--output", "cli-runs/out")
        self.assertEqual(project.runs_dir, self.root / "process-runs")
        self.assertEqual(args.hf_repo, "cli/yolo")
        self.assertEqual(args.device, 4)
        self.assertEqual(args.python, self.root / "cli-python")
        self.assertEqual(args.output, self.root / "cli-runs/out")

    def test_model_specific_flags_are_rejected(self):
        bad = (("evaluate", ("yolo26", "--checkpoint", "a.pt", "--max-new-tokens", "9")),
               ("evaluate", ("medgemma", "--checkpoint", "a", "--imgsz", "960")),
               ("evaluate", ("medgemma", "--checkpoint", "a", "--batch", "4")),
               ("evaluate", ("medgemma", "--checkpoint", "a", "--hf-filename", "x.pt")),
               ("evaluate", ("yolo26", "--checkpoint", "a.pt", "--hf-run", "run")))
        for mode, argv in bad:
            with self.subTest(argv=argv), self.assertRaisesRegex(ValueError, "YOLO-only|MedGemma-only"):
                self.resolve(mode, *argv)

    def test_local_hf_exclusivity_and_missing_repo_are_execution_guards(self):
        with self.assertRaises(SystemExit):
            self.resolve("evaluate", "yolo26", "--checkpoint", "a.pt", "--hf")
        with self.assertRaisesRegex(ValueError, "Use --hf"):
            self.resolve("evaluate", "yolo26", "--checkpoint", "a.pt", "--hf-repo", "x/y")
        _, _, plan = self.plan("evaluate", "yolo26", "--hf", "--hf-filename", "model.pt")
        self.assertIn("--hf-repo or configured model repository", plan["missing_for_execution"])
        _, _, plan = self.plan("evaluate", "yolo26", "--checkpoint", "a.pt")
        self.assertNotIn("--hf-repo or configured model repository", plan["missing_for_execution"])

    def test_preview_is_offline_and_does_not_import_ml_or_spawn(self):
        args, project = self.resolve("evaluate", "yolo26", "--checkpoint", "a.pt", "--print-config")
        with patch.object(cli.subprocess, "run", side_effect=AssertionError("preview spawned")), \
             patch.dict(sys.modules, {name: None for name in ("torch", "numpy", "pandas", "cv2", "huggingface_hub", "requests")}):
            plan = cli.make_plan(args, project)
            self.assertEqual(plan["model"], "yolo26")
        self.assertNotIn("torch", cli.__dict__)

    def test_print_config_works_in_fresh_no_site_child(self):
        env = os.environ.copy()
        env.pop("PYTHONHOME", None)
        env.pop("PYTHONPATH", None)
        env["PYTHONHOME"] = sys.base_prefix
        result = subprocess.run([sys.executable, "-B", "-S", str(ROOT / "evaluate.py"), "yolo26", "--project-root", str(self.root),
                                 "--checkpoint", "missing.pt", "--print-config"], cwd=str(self.root), env=env,
                                text=True, capture_output=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["checkpoint"], str(self.root / "missing.pt"))

    def test_checkpoint_and_python_symlink_names_are_preserved(self):
        target = self.root / "real.pt"
        target.write_text("weights", encoding="utf-8")
        checkpoint = self.root / "cache-name.pt"
        checkpoint.symlink_to(target)
        py_target = Path(sys.executable)
        python = self.root / "venv-python"
        python.symlink_to(py_target)
        args, _, plan = self.plan("evaluate", "yolo26", "--checkpoint", str(checkpoint), "--python", str(python))
        self.assertEqual(args.checkpoint, checkpoint)
        self.assertEqual(args.python, python)
        self.assertEqual(plan["command"][0], str(python))

    def test_both_models_default_to_yolo_dataset(self):
        for model in ("yolo26", "medgemma"):
            with self.subTest(model=model):
                args, project = self.resolve("evaluate", model, "--checkpoint", "model")
                self.assertEqual(args.dataset, project.yolo_data_dir)
                self.assertNotEqual(args.dataset, project.medgemma_data_dir)

    def test_default_output_is_a_new_runs_directory(self):
        self.write_env("RUNS_DIR=portable-runs\n")
        args, project = self.resolve("evaluate", "yolo26", "--checkpoint", "a.pt")
        self.assertEqual(args.output.parent, project.runs_dir)
        self.assertTrue(args.output.name.startswith("evaluation_yolo26_val_"))
        self.assertFalse(os.path.lexists(args.output))

    def test_new_output_and_source_safety_guards(self):
        dataset = self.root / "data/grazpedwri_yolo"
        output = dataset / "bad-output"
        with self.assertRaisesRegex(ValueError, "outside the prepared dataset"):
            self.resolve("evaluate", "yolo26", "--checkpoint", "a.pt", "--output", str(output))
        source = self.root / "images"
        source.mkdir()
        with self.assertRaisesRegex(ValueError, "outside the source"):
            self.resolve("infer", "yolo26", "--checkpoint", "a.pt", "--source", str(source), "--conf", "0.2", "--output", str(source / "out"))
        existing = self.root / "existing"
        existing.mkdir()
        with self.assertRaisesRegex(ValueError, "Output already exists"):
            cli.main("evaluate", ["--project-root", str(self.root), "yolo26", "--checkpoint", "a.pt", "--python", sys.executable, "--output", str(existing)])

    def test_test_requires_complete_matching_validation_protocol(self):
        with self.assertRaisesRegex(ValueError, "Only test requires"):
            self.resolve("evaluate", "yolo26", "--checkpoint", "a.pt", "--split", "test")
        path, report = self.protocol(model="yolo26")
        args, project = self.resolve("evaluate", "yolo26", "--checkpoint", "a.pt", "--split", "test", "--protocol", str(path))
        self.assertEqual(cli.read_protocol(args), report)
        for change in ({"status": "running"}, {"split": "test"}, {"protocol__identity__model": "medgemma"},
                       {"protocol__identity__raw_manifest_sha256": "0" * 64}):
            path, _ = self.protocol(model="yolo26", **change)
            args, _ = self.resolve("evaluate", "yolo26", "--checkpoint", "a.pt", "--split", "test", "--protocol", str(path))
            with self.subTest(change=change), self.assertRaises(ValueError):
                cli.read_protocol(args)

    def test_yolo_protocol_inherits_accepted_values_and_rejects_mismatch(self):
        path, _ = self.protocol(model="yolo26")
        _, _, plan = self.plan("evaluate", "yolo26", "--checkpoint", "a.pt", "--split", "test", "--protocol", str(path))
        self.assertEqual(plan["prediction"], {"imgsz": 960, "batch": 4, "max_new_tokens": None})
        _, _, plan = self.plan("evaluate", "yolo26", "--checkpoint", "a.pt", "--split", "test", "--protocol", str(path), "--imgsz", "960", "--batch", "4")
        self.assertEqual(plan["prediction"]["imgsz"], 960)
        with self.assertRaisesRegex(ValueError, "differs from validation"):
            self.plan("evaluate", "yolo26", "--checkpoint", "a.pt", "--split", "test", "--protocol", str(path), "--batch", "8")

    def test_medgemma_protocol_inherits_max_new_tokens(self):
        path, _ = self.protocol(model="medgemma")
        _, _, plan = self.plan("evaluate", "medgemma", "--checkpoint", "model", "--split", "test", "--protocol", str(path))
        self.assertEqual(plan["prediction"], {"imgsz": None, "batch": None, "max_new_tokens": 777})

    def test_frozen_child_and_metadata_hashes_are_planned(self):
        _, _, plan = self.plan("evaluate", "medgemma", "--checkpoint", "model")
        self.assertEqual(plan["command"][2], str(ROOT / "evaluation_scripts/evaluate_medgemma.py"))
        self.assertEqual(set(plan["backend_source_sha256"]), {"evaluate_medgemma.py", "evaluation_common.py", "checkpoints.py", "medgemma_model.py"})
        self.assertEqual(set(plan["launcher_source_sha256"]), {"evaluate.py", "prediction_cli.py", "config.py"})

    def test_source_mismatch_is_rejected_without_changing_report(self):
        path, original = self.protocol(model="yolo26", protocol__identity__source_sha256={"evaluate_yolo26.py": "bad"})
        before = path.read_bytes()
        args, _ = self.resolve("evaluate", "yolo26", "--checkpoint", "a.pt", "--split", "test", "--protocol", str(path))
        with self.assertRaisesRegex(ValueError, "Source differs"):
            cli.read_protocol(args)
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(json.loads(before), original)

    def test_infer_requires_protocol_or_explicit_exploratory_conf_range(self):
        _, _, incomplete = self.plan("infer", "yolo26", "--checkpoint", "a.pt", "--source", "image.png")
        self.assertIn("--protocol or explicit exploratory --conf", incomplete["missing_for_execution"])
        for bad in ("0", "0.0009", "1.1", "nan", "inf"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.resolve("infer", "yolo26", "--checkpoint", "a.pt", "--source", "image.png", "--conf", bad)
        _, _, plan = self.plan("infer", "yolo26", "--checkpoint", "a.pt", "--source", "image.png", "--conf", "0.2")
        self.assertEqual(plan["confidence_cutoff"], 0.2)

    def test_mocked_child_receipt_and_environment_do_not_serialize_secrets(self):
        self.write_env("HF_TOKEN=dotenv-secret\nHF_HOME=hub-cache\nCUDA_VISIBLE_DEVICES=7\n")
        output = self.root / "success"
        captured = {}
        def fake_run(command, cwd, env, check):
            captured.update(command=command, cwd=cwd, env=env.copy(), check=check)
            output.mkdir()
            return SimpleNamespace(returncode=0)
        with patch.object(cli.subprocess, "run", side_effect=fake_run):
            code = cli.main("evaluate", ["--project-root", str(self.root), "yolo26", "--checkpoint", "a.pt", "--python", sys.executable, "--output", str(output)])
        self.assertEqual(code, 0)
        self.assertFalse(captured["check"])
        self.assertEqual(captured["env"]["CUDA_VISIBLE_DEVICES"], "7")
        self.assertEqual(captured["env"]["YOLO_AUTOINSTALL"], "false")
        receipt = (output / "launcher_config.json").read_text(encoding="utf-8")
        self.assertNotIn("dotenv-secret", receipt)
        self.assertNotIn("HF_TOKEN", receipt)

    def test_receipt_only_after_successful_new_output_and_native_report_is_unchanged(self):
        output = self.root / "native"
        native = {"status": "complete", "native": True}
        def successful(*_args, **_kwargs):
            output.mkdir()
            (output / "evaluation.json").write_text(json.dumps(native), encoding="utf-8")
            return SimpleNamespace(returncode=0)
        with patch.object(cli.subprocess, "run", side_effect=successful):
            self.assertEqual(cli.main("evaluate", ["--project-root", str(self.root), "yolo26", "--checkpoint", "a.pt", "--python", sys.executable, "--output", str(output)]), 0)
        self.assertEqual(json.loads((output / "evaluation.json").read_text(encoding="utf-8")), native)
        self.assertTrue((output / "launcher_config.json").is_file())
        failed = self.root / "failed"
        with patch.object(cli.subprocess, "run", return_value=SimpleNamespace(returncode=9)):
            self.assertEqual(cli.main("evaluate", ["--project-root", str(self.root), "yolo26", "--checkpoint", "a.pt", "--python", sys.executable, "--output", str(failed)]), 9)
        self.assertFalse((failed / "launcher_config.json").exists())


if __name__ == "__main__":
    unittest.main()
