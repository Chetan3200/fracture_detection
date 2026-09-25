"""CPU-only backup tests. Uses fake Hub APIs; never logs in, uploads, or loads torch."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import ast
import hashlib
import importlib.util
import io
import json
import tempfile
import unittest
import zipfile

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SCRIPT = ROOT / "train_medgemma.py"
SUPPORT = ROOT / "medgemma_support.py"
spec = importlib.util.spec_from_file_location("medgemma_backup_under_test", SUPPORT)
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)


class Add:
    def __init__(self, path_in_repo, path_or_fileobj):
        self.path_in_repo = path_in_repo
        self.path_or_fileobj = path_or_fileobj


class API:
    def __init__(self, private=True, exists=False, failures=0):
        self.private, self.exists, self.failures = private, exists, failures
        self.attempts, self.blobs, self.create_options = [], {}, None
    def whoami(self):
        return {"name": "test-account"}
    def create_repo(self, **kw):
        self.create_options = kw
    def repo_info(self, **kw):
        return SimpleNamespace(private=self.private)
    def file_exists(self, **kw):
        return self.exists
    def create_commit(self, **kw):
        self.attempts.append(kw)
        if len(self.attempts) <= self.failures:
            raise ConnectionError("simulated network failure")
        for op in kw["operations"]:
            val = op.path_or_fileobj
            self.blobs[op.path_in_repo] = val if isinstance(val, bytes) else Path(val).read_bytes()
        return SimpleNamespace(oid="test-commit", commit_url="https://huggingface.co/test/backup/commit/test-commit")


def fixture(parent, world=2, best=True, config_sha256=False):
    output = Path(parent) / "test-run"
    output.mkdir()
    for name in ["run_config.json", "prompt.txt", "frozen_split_manifest.csv",
                 "preprocessing_and_lora_audit.json", "training_script.py", "requirements_medgemma.txt"]:
        (output / name).write_text("{}")
    run_config = {"original_output_dir": str(output.resolve()), "run_name": output.name}
    if config_sha256:
        config_bytes = b"SNAPSHOTTED_CONFIG = 'verified'\n"
        (output / "config.py").write_bytes(config_bytes)
        run_config["config_sha256"] = hashlib.sha256(config_bytes).hexdigest()
    (output / "run_config.json").write_text(json.dumps(run_config))
    (output / "processor").mkdir()
    (output / "processor" / "tokenizer_config.json").write_text("{}")

    def checkpoint(step, selected):
        folder = output / f"checkpoint-{step}"
        folder.mkdir()
        for name in ["training_args.bin", "adapter_config.json", "adapter_model.safetensors", "optimizer.pt", "scheduler.pt"]:
            (folder / name).write_bytes(b"nonempty mocked state")
        names = [f"rng_state_{rank}.pth" for rank in range(world)] if world > 1 else ["rng_state.pth"]
        for name in names:
            (folder / name).write_bytes(b"mocked RNG")
        (folder / "trainer_state.json").write_text(json.dumps({"global_step": step, "best_model_checkpoint": selected}))
        return folder

    selected = output / "checkpoint-8" if best else None
    if best:
        checkpoint(8, str(selected))
    latest = checkpoint(12, str(selected) if best else None)
    return output, latest


class Tests(unittest.TestCase):
    def test_divisors_preserve_effective_batch(self):
        for world, accum in [(1, 16), (2, 8), (4, 4), (8, 2), (16, 1)]:
            self.assertEqual(app.default_accumulation(world), accum)
            self.assertEqual(world * accum, 16)

    def test_nondivisor_requires_explicit_accumulation(self):
        for world in [3, 5, 6, 17, 32]:
            with self.assertRaisesRegex(ValueError, "set --grad-accum explicitly"):
                app.default_accumulation(world)

    def test_world_size_must_be_a_positive_plain_integer(self):
        for world in [0, -1, True, False, 1.0, 2.5]:
            with self.assertRaisesRegex(ValueError, "positive integer GPU/rank count"):
                app.validate_world_size(world)
            with self.assertRaisesRegex(ValueError, "positive integer GPU/rank count"):
                app.HFCheckpointBackup(API(), Add, "test/backup", ".", world)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "positive integer GPU/rank count"):
                app.validate_checkpoint(Path(tmp), tmp, 0)

    def test_resume_requires_original_absolute_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            original = Path(tmp) / "a" / "run"
            moved = Path(tmp) / "b" / "run"
            config = {"original_output_dir": str(original.resolve())}
            app.validate_resume_run_location(config, original)
            with self.assertRaisesRegex(ValueError, "original absolute"):
                app.validate_resume_run_location(config, moved)
            with self.assertRaises(ValueError):
                app.validate_resume_run_location({}, original)

    def test_repository_privacy_is_rechecked_before_every_backup(self):
        with tempfile.TemporaryDirectory() as tmp:
            output, latest = fixture(tmp)
            api = API()
            uploader = app.HFCheckpointBackup(api, Add, "test/backup", output, 2)
            uploader.initialize()
            api.private = False
            with self.assertRaisesRegex(ValueError, "no longer private"):
                uploader.upload(latest)
            self.assertEqual(len(api.attempts), 1)

    def test_periodic_optimizer_step_schedule(self):
        self.assertTrue(app.periodic_checkpoint_due(100, 100, 100 / 840))
        self.assertTrue(app.periodic_checkpoint_due(100, 100, None))
        for args in [(0, 100, 0), (99, 100, .1), (100, 0, .1), (100, 100, 1.0), (840, 20, 1.0)]:
            self.assertFalse(app.periodic_checkpoint_due(*args))

    def test_multirank_rng_completeness_and_fake_hub_uploads(self):
        for world in [3, 5, 8, 17]:
            with self.subTest(world=world), tempfile.TemporaryDirectory() as tmp:
                output, latest = fixture(tmp, world=world)
                self.assertEqual(app.validate_checkpoint(latest, output, world), latest.resolve())
                api = API()
                receipt = app.HFCheckpointBackup(api, Add, "test/backup", output, world).upload(latest)
                with zipfile.ZipFile(io.BytesIO(api.blobs[receipt["archive_path"]])) as archive:
                    self.assertIn(f"checkpoint-12/rng_state_{world - 1}.pth", archive.namelist())
                (latest / f"rng_state_{world - 1}.pth").unlink()
                with self.assertRaisesRegex(ValueError, f"rng_state_{world - 1}\\.pth"):
                    app.backup_payload(output, latest, world)

    def test_single_rank_uses_unranked_rng_filename(self):
        with tempfile.TemporaryDirectory() as tmp:
            output, latest = fixture(tmp, world=1)
            self.assertEqual(app.validate_checkpoint(latest, output, 1), latest.resolve())
            (latest / "rng_state.pth").unlink()
            with self.assertRaisesRegex(ValueError, "rng_state\\.pth"):
                app.backup_payload(output, latest, 1)

    def test_latest_and_best_are_both_recoverable(self):
        with tempfile.TemporaryDirectory() as tmp:
            output, latest = fixture(tmp)
            files, name, best = app.backup_payload(output, latest, 2)
            paths = {p.relative_to(output).as_posix() for p in files}
            self.assertEqual((name, best), ("checkpoint-12", "checkpoint-8"))
            for ck in (name, best):
                for f in ("optimizer.pt", "scheduler.pt", "rng_state_0.pth", "rng_state_1.pth", "trainer_state.json"):
                    self.assertIn(f"{ck}/{f}", paths)

    def test_pre_first_validation_snapshot_has_no_best(self):
        with tempfile.TemporaryDirectory() as tmp:
            output, latest = fixture(tmp, best=False)
            files, _, best = app.backup_payload(output, latest, 2)
            self.assertIsNone(best)
            self.assertTrue(files)

    def test_missing_best_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            output, latest = fixture(tmp)
            (output / "checkpoint-8" / "optimizer.pt").unlink()
            with self.assertRaisesRegex(ValueError, "optimizer.pt"):
                app.backup_payload(output, latest, 2)

    def test_allowlist_excludes_dataset_and_credentials(self):
        with tempfile.TemporaryDirectory() as tmp:
            output, latest = fixture(tmp)
            for name in ["token", ".env", "test_inputs.jsonl"]:
                (output / name).write_text("must not upload")
            (output / "dataset").mkdir()
            (output / "dataset" / "image.png").write_bytes(b"must not upload")
            files, _, _ = app.backup_payload(output, latest, 2)
            paths = {p.name for p in files}
            self.assertTrue(paths.isdisjoint({"token", ".env", "test_inputs.jsonl", "image.png"}))
            api = API()
            receipt = app.HFCheckpointBackup(api, Add, "test/backup", output, 2).upload(latest)
            with zipfile.ZipFile(io.BytesIO(api.blobs[receipt["archive_path"]])) as z:
                archived = set(z.namelist())
            self.assertTrue(archived.isdisjoint({"token", ".env", "test_inputs.jsonl", "dataset/image.png"}))

    def test_verified_config_snapshot_is_in_recovery_zip(self):
        with tempfile.TemporaryDirectory() as tmp:
            output, latest = fixture(tmp, config_sha256=True)
            config_bytes = (output / "config.py").read_bytes()
            api = API()
            receipt = app.HFCheckpointBackup(api, Add, "test/backup", output, 2).upload(latest)
            with zipfile.ZipFile(io.BytesIO(api.blobs[receipt["archive_path"]])) as z:
                self.assertEqual(z.read("config.py"), config_bytes)
                manifest = json.loads(z.read("backup_manifest.json"))
                self.assertIn("config.py", manifest["files"])

    def test_missing_verified_config_snapshot_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            output, latest = fixture(tmp, config_sha256=True)
            (output / "config.py").unlink()
            with self.assertRaisesRegex(ValueError, "Training config is missing or changed"):
                app.backup_payload(output, latest, 2)

    def test_changed_verified_config_snapshot_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            output, latest = fixture(tmp, config_sha256=True)
            (output / "config.py").write_text("CHANGED = True\n")
            with self.assertRaisesRegex(ValueError, "Training config is missing or changed"):
                app.backup_payload(output, latest, 2)

    def test_symlinked_verified_config_snapshot_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            output, latest = fixture(tmp, config_sha256=True)
            (output / "config.py").unlink()
            (output / "config.py").symlink_to(output / "training_script.py")
            with self.assertRaisesRegex(ValueError, "Training config is missing or changed"):
                app.backup_payload(output, latest, 2)

    def test_older_metadata_optionally_includes_config_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            output, latest = fixture(tmp)
            config_bytes = b"LEGACY_CONFIG = True\n"
            (output / "config.py").write_bytes(config_bytes)
            files, _, _ = app.backup_payload(output, latest, 2)
            self.assertIn(output / "config.py", files)
            api = API()
            receipt = app.HFCheckpointBackup(api, Add, "test/backup", output, 2).upload(latest)
            with zipfile.ZipFile(io.BytesIO(api.blobs[receipt["archive_path"]])) as z:
                self.assertEqual(z.read("config.py"), config_bytes)

    def test_symlink_in_processor_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            output, latest = fixture(tmp)
            (output / "processor" / "link").symlink_to(output / "prompt.txt")
            with self.assertRaisesRegex(ValueError, "symlink"):
                app.backup_payload(output, latest, 2)

    def test_public_repository_fails_before_commit(self):
        with tempfile.TemporaryDirectory() as tmp:
            output, _ = fixture(tmp)
            api = API(private=False)
            uploader = app.HFCheckpointBackup(api, Add, "test/backup", output, 2)
            with self.assertRaisesRegex(ValueError, "public"):
                uploader.initialize()
            self.assertEqual(api.attempts, [])
            self.assertTrue(api.create_options["private"])

    def test_remote_name_collision_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            output, _ = fixture(tmp)
            api = API(exists=True)
            uploader = app.HFCheckpointBackup(api, Add, "test/backup", output, 2)
            with self.assertRaisesRegex(ValueError, "already exists"):
                uploader.initialize()
            uploader.initialize(resume=True)
            self.assertEqual(len(api.attempts), 1)

    def test_atomic_archive_pointer_and_checksum(self):
        with tempfile.TemporaryDirectory() as tmp:
            output, latest = fixture(tmp)
            api = API()
            uploader = app.HFCheckpointBackup(api, Add, "test/backup", output, 2)
            receipt = uploader.upload(latest)
            self.assertEqual(len(api.attempts), 1)
            self.assertEqual(len(api.attempts[0]["operations"]), 3)
            pointer = json.loads(api.blobs["runs/test-run/latest.json"])
            archive = api.blobs[pointer["archive_path"]]
            self.assertEqual(hashlib.sha256(archive).hexdigest(), pointer["archive_sha256"])
            with zipfile.ZipFile(io.BytesIO(archive)) as z:
                self.assertIsNone(z.testzip())
                self.assertIn("checkpoint-12/rng_state_1.pth", z.namelist())
                self.assertIn("checkpoint-8/optimizer.pt", z.namelist())
                self.assertIn("backup_manifest.json", z.namelist())
            self.assertEqual(receipt["commit_sha"], "test-commit")
            self.assertTrue((output / "hf_backup_latest.json").is_file())
            self.assertEqual(list(Path(tmp).glob("hf-backup-*")), [])

    def test_recovery_archive_roundtrip_preserves_resume_paths_and_rank_states(self):
        with tempfile.TemporaryDirectory() as tmp:
            output, latest = fixture(tmp)
            api = API()
            receipt = app.HFCheckpointBackup(api, Add, "test/backup", output, 2).upload(latest)
            archive = api.blobs[receipt["archive_path"]]
            output.rename(Path(tmp) / "old-run")
            with zipfile.ZipFile(io.BytesIO(archive)) as z:
                z.extractall(output)  # Only our own synthetic, allowlisted archive.
            config = json.loads((output / "run_config.json").read_text())
            app.validate_resume_run_location(config, output)
            for name in (receipt["checkpoint"], receipt["best_checkpoint"]):
                app.validate_checkpoint(output / name, output, 2)
            state = json.loads((output / receipt["checkpoint"] / "trainer_state.json").read_text())
            self.assertEqual(state["best_model_checkpoint"], str(output / receipt["best_checkpoint"]))

    def test_retries_build_fresh_operations(self):
        with tempfile.TemporaryDirectory() as tmp:
            output, latest = fixture(tmp)
            api, delays = API(failures=2), []
            uploader = app.HFCheckpointBackup(api, Add, "test/backup", output, 2, sleep_fn=delays.append)
            uploader.upload(latest)
            self.assertEqual(delays, [15, 60])
            self.assertEqual(len(api.attempts), 3)
            first_ops = [a["operations"][0] for a in api.attempts]
            self.assertEqual(len({id(op) for op in first_ops}), 3)

    def test_failed_upload_preserves_checkpoint_and_no_success_pointer(self):
        with tempfile.TemporaryDirectory() as tmp:
            output, latest = fixture(tmp)
            api = API(failures=10)
            uploader = app.HFCheckpointBackup(api, Add, "test/backup", output, 2, sleep_fn=lambda _: None)
            with self.assertRaisesRegex(RuntimeError, "Training stops"):
                uploader.upload(latest)
            self.assertTrue((latest / "optimizer.pt").is_file())
            self.assertFalse((output / "hf_backup_latest.json").exists())
            self.assertEqual(api.blobs, {})
            self.assertEqual(list(Path(tmp).glob("hf-backup-*")), [])

    def test_final_archive_contains_selected_adapter_and_generation_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            output, latest = fixture(tmp)
            for name in ["training_summary.json", "gpu_memory.json", "validation_format_check.jsonl", "validation_format_summary.json"]:
                (output / name).write_text("{}")
            (output / "best_adapter").mkdir()
            for name in ["adapter_config.json", "adapter_model.safetensors"]:
                (output / "best_adapter" / name).write_text("nonempty mock")
            api = API()
            result = app.HFCheckpointBackup(api, Add, "test/backup", output, 2).upload(latest, final=True)
            with zipfile.ZipFile(io.BytesIO(api.blobs[result["archive_path"]])) as z:
                self.assertIn("best_adapter/adapter_model.safetensors", z.namelist())
                self.assertIn("validation_format_summary.json", z.namelist())
            self.assertEqual(result["kind"], "final")

    def test_callback_has_pre_and_post_collective_guards(self):
        tree = ast.parse(SCRIPT.read_text())
        callback = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "on_save")
        guarded = callback.body[0]
        self.assertIsInstance(guarded, ast.If)
        calls = [n.value.func.id for n in guarded.body if isinstance(n, ast.Expr)
                 and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name)]
        self.assertEqual(calls, ["barrier", "primary_call", "barrier"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
