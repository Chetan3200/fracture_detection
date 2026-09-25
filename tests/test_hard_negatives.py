"""Offline regression checks for the frozen hard-negative YOLO experiment.

Run with: python -B -m unittest tests/test_hard_negatives.py -v
No image is read and no real ML, Hub, network, or training dependency is used.
"""
from collections import Counter, defaultdict
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from types import ModuleType, SimpleNamespace
import csv
import hashlib
import io
import json
import os
import random
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config
import train_hard_negatives as hn


class Fixture(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=os.environ.get("FRACTURE_TEST_TMP"))
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.environment = dict(os.environ)
        self.env_patch = patch.dict(os.environ, {}, clear=True)
        self.env_patch.start()
        self.addCleanup(self.env_patch.stop)
        self.manifest_bytes = (ROOT / "split_manifest.csv").read_bytes()
        self.candidate_bytes = (ROOT / "hard_negatives_review/ranked_negatives.csv").read_bytes()

    def argv(self, *items):
        return ["--project-root", str(self.root), *items]

    def prepared_dataset(self):
        data = self.root / "data/grazpedwri_yolo"
        (data / "labels/train").mkdir(parents=True)
        (data / "split_manifest.csv").write_bytes(self.manifest_bytes)
        (self.root / "split_manifest.csv").write_bytes(self.manifest_bytes)
        (data / "dataset_info.json").write_text(json.dumps({"manifest_sha256": config.MANIFEST_SHA256}))
        (data / "data.yaml").write_text(json.dumps({"train": "images/train", "val": "images/val", "names": ["fracture"]}))
        return data

    def frozen_pool(self, stack, candidate_bytes=None):
        data = self.prepared_dataset()
        original = Path.read_text
        def only_synthetic_labels(path, *args, **kwargs):
            if path.parent == data / "labels/train" and path.suffix == ".txt":
                return ""
            return original(path, *args, **kwargs)
        stack.enter_context(patch.object(Path, "read_text", only_synthetic_labels))
        return hn.load_pool(candidate_bytes or self.candidate_bytes, data, self.root / "split_manifest.csv")


class FrozenInputTests(Fixture):
    def test_frozen_hashes_recipe_and_matched_batches(self):
        self.assertEqual(config.MANIFEST_SHA256, "1719f37f442512c3c4fcab8350bedc0f27ff2dcebc082761299912b897747034")
        self.assertEqual(config.CANDIDATE_CSV_SHA256, "5af4772bf275a4e84a815670d2988fb0408119ea05e0b2ffe0cb3251fdf1e66d")
        self.assertEqual(config.HARD_NEGATIVE_POOL_SHA256, "2db84c52f39debc56ad6d6bdae9428ef8d6f2589e7e208781bcaa5f6ddf1a719")
        self.assertEqual((config.YOLO_BATCH_SIZES, config.YOLO_EPOCHS, config.YOLO_SEED, config.HARD_NEGATIVE_WEIGHT),
                         ({640: 35, 960: 14}, 100, 43, 3.0))

    def test_tracked_inputs_select_exact_capped_pool_and_reject_corruption(self):
        with ExitStack() as stack:
            cohort, pool, normalized = self.frozen_pool(stack)
            self.assertEqual(Counter(v[1] for v in cohort.values()), {True: 9479, False: 3961})
            self.assertEqual(len(pool), 178)
            self.assertEqual(hashlib.sha256(normalized).hexdigest(), config.HARD_NEGATIVE_POOL_SHA256)
            self.assertLessEqual(max(Counter(row["patient_id"] for row in pool).values()), 2)
            self.assertTrue(all(row["review"] in {"annotation_defined", "keep"} for row in pool))
            changed = self.candidate_bytes.replace(b",yes,,,", b",yes,exclude,,", 1)
            with self.assertRaisesRegex(RuntimeError, "exactly 178"):
                hn.load_pool(changed, self.root / "data/grazpedwri_yolo", self.root / "split_manifest.csv")
            bad_score = self.candidate_bytes.replace(b"0.758301", b"nan", 1)
            with self.assertRaisesRegex(RuntimeError, "mining floor"):
                hn.load_pool(bad_score, self.root / "data/grazpedwri_yolo", self.root / "split_manifest.csv")

    def test_alternate_candidate_representation_needs_explicit_hash_and_same_pool(self):
        source = list(csv.DictReader(io.StringIO(self.candidate_bytes.decode("utf-8-sig"))))
        text = io.StringIO(newline="")
        fields = ["image_id", "patient_id", "hardness_score", "candidate_status", "review_decision"]
        writer = csv.DictWriter(text, fields); writer.writeheader()
        for row in source:
            writer.writerow({"image_id": row["image_id"], "patient_id": row["patient_id"],
                             "hardness_score": row["hardness_score"],
                             "candidate_status": "candidate_pending_review" if row["candidate"] == "yes" else "not_candidate",
                             "review_decision": row["review"]})
        alternate = text.getvalue().encode()
        candidate = self.root / "alternate.csv"; candidate.write_bytes(alternate)
        explicit = hashlib.sha256(alternate).hexdigest()
        args, _ = hn.resolve_arguments(self.argv("--imgsz", "640", "--print-config", "--candidates", str(candidate),
                                                   "--candidate-sha256", explicit))
        self.assertEqual(args.candidate_sha256, explicit)
        with ExitStack() as stack:
            _, _, normalized = self.frozen_pool(stack, alternate)
        self.assertEqual(hashlib.sha256(normalized).hexdigest(), config.HARD_NEGATIVE_POOL_SHA256)


class SamplerTests(Fixture):
    def test_primary_sampler_has_exact_draws_determinism_and_local_rng(self):
        cohort = {"p%05d" % n: (n, True) for n in range(9479)}
        cohort.update({"n%05d" % n: (100000 + n, False) for n in range(3961)})
        hard = {"n%05d" % n for n in range(178)}
        reverse_ids = list(reversed(list(cohort)))
        first, second = hn.PrimarySampler(reverse_ids, cohort, hard, 43), hn.PrimarySampler(list(cohort), cohort, hard, 43)
        random.seed(819); before = random.getstate()
        first.set_epoch(0); after = random.getstate()
        second.set_epoch(0)
        self.assertEqual(before, after, "sampling must not consume process-global RNG")
        ids = [first.ids[index] for index in first]
        self.assertEqual(ids, [second.ids[index] for index in second], "loader order must not alter the epoch plan")
        draws = Counter(ids)
        self.assertEqual(len(ids), 13440)
        self.assertTrue(all(draws["p%05d" % n] == 1 for n in range(9479)))
        self.assertEqual(sum(draws["n%05d" % n] for n in range(3961)), 3961)
        first.set_epoch(1); epoch_one = [first.ids[index] for index in first]
        first.set_epoch(1); self.assertEqual(epoch_one, [first.ids[index] for index in first])
        self.assertNotEqual(epoch_one, ids)


class ArgumentsAndOfflineCliTests(Fixture):
    def test_env_process_cli_precedence_paths_and_no_secret_print(self):
        secret = "not-a-real-secret"
        (self.root / ".env").write_text("DATA_DIR=dotenv-data\nRUNS_DIR=dotenv-runs\nHF_REPO_ID=owner/dotenv\nHF_TOKEN=%s\n" % secret)
        os.environ.update(DATA_DIR="process-data", HF_REPO_ID="owner/process")
        args, project = hn.resolve_arguments(self.argv("--imgsz", "960", "--print-config", "--dataset", "cli-data",
                                                        "--output-root", "cli-runs", "--repo-id", "owner/cli"))
        value = hn.resolved_configuration(args, project)
        serialized = json.dumps(value)
        self.assertEqual((args.dataset, args.output_root, args.repo_id),
                         (self.root / "cli-data", self.root / "cli-runs", "owner/cli"))
        self.assertEqual(project.data_dir, self.root / "process-data")
        self.assertNotIn(secret, serialized); self.assertNotIn("HF_TOKEN", serialized)

    def test_recovery_controls_require_pinned_boundary_complete_options_and_frozen_pool(self):
        base = self.argv("--imgsz", "640", "--repo-id", "owner/repo")
        for extra in (("--resume-hf-run", "run"), ("--resume-epoch", "10"),
                      ("--resume-hf-run", "run", "--resume-epoch", "11", "--hf-revision", "a" * 40),
                      ("--resume-hf-run", "run", "--resume-epoch", "10", "--hf-revision", "a" * 39),
                      ("--resume-hf-run", "run", "--resume-epoch", "10", "--hf-revision", "a" * 40, "--candidates", "other.csv")):
            with self.subTest(extra=extra), self.assertRaises((RuntimeError, ValueError)):
                hn.resolve_arguments([*base, *extra])
        args, _ = hn.resolve_arguments([*base, "--resume-hf-run", "run_10", "--resume-epoch", "10", "--hf-revision", "A" * 40])
        self.assertEqual((args.candidates, args.candidate_sha256, args.hf_revision), (None, None, "a" * 40))

    def test_help_and_print_config_work_from_foreign_cwd_with_poison_modules(self):
        poison = self.root / "poison"; poison.mkdir()
        for name in ("torch", "ultralytics", "huggingface_hub"):
            (poison / (name + ".py")).write_text("raise RuntimeError('poison optional import')\n")
        env = self.environment.copy(); env["PYTHONPATH"] = str(poison) + os.pathsep + env.get("PYTHONPATH", "")
        foreign = self.root / "foreign"; foreign.mkdir()
        commands = (["--help"], ["--project-root", str(self.root), "--imgsz", "960", "--print-config"])
        for command in commands:
            result = subprocess.run([sys.executable, "-B", str(ROOT / "train_hard_negatives.py"), *command], cwd=foreign,
                                    env=env, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            if "--print-config" in command:
                self.assertEqual(json.loads(result.stdout)["batch_size"], 14)


class ReadinessAndMockedFlowTests(Fixture):
    def test_bad_candidate_is_rejected_before_ml_or_hub_import(self):
        data = self.prepared_dataset()
        bad = self.root / "bad.csv"; bad.write_bytes(b"not the frozen csv")
        args, project = hn.resolve_arguments(self.argv("--imgsz", "640", "--repo-id", "owner/repo", "--candidates", str(bad),
                                                        "--candidate-sha256", hashlib.sha256(self.candidate_bytes).hexdigest()))
        imports = []
        original = __import__("builtins").__import__
        def guarded(name, *items, **kwargs):
            imports.append(name)
            if name.split(".")[0] in {"torch", "ultralytics", "huggingface_hub"}:
                raise AssertionError("ML/Hub import occurred before input readiness")
            return original(name, *items, **kwargs)
        with patch("builtins.__import__", guarded), self.assertRaisesRegex(RuntimeError, "Candidate CSV changed"):
            hn.run_training(args, project, hn.resolved_configuration(args, project))
        self.assertFalse(set(imports) & {"torch", "ultralytics", "huggingface_hub"})
        self.assertEqual(data, self.root / "data/grazpedwri_yolo")

    def test_mocked_fresh_main_uses_one_100_epoch_call_and_ten_epoch_snapshots(self):
        data = self.prepared_dataset(); start_cwd = Path.cwd()
        candidates = self.root / "hard_negatives_review/ranked_negatives.csv"
        candidates.parent.mkdir(); candidates.write_bytes(self.candidate_bytes)
        state = SimpleNamespace(commits=[], train_calls=[], writes=0)
        class API:
            def whoami(self): return {"name": "fixture"}
            def create_repo(self, *args, **kwargs): state.created = kwargs
            def repo_info(self, *args, **kwargs): return SimpleNamespace(private=True)
            def file_exists(self, *args, **kwargs): return False
            def create_commit(self, **kwargs): state.commits.append(kwargs); return SimpleNamespace(commit_url="https://example.invalid/commit")
        class YOLO:
            def __init__(self, checkpoint): self.ckpt_path = ROOT / "train_hard_negatives.py"; self.callbacks = defaultdict(list)
            def add_callback(self, name, function): self.callbacks[name].append(function)
            def train(self, **kwargs):
                state.train_calls.append(kwargs)
                run = Path(kwargs["project"]) / kwargs["name"]; (run / "weights").mkdir(exist_ok=True)
                trainer = SimpleNamespace(save_dir=run, start_epoch=0, epochs=kwargs["epochs"], batch_size=kwargs["batch"], amp=True, epoch=0,
                                          last=run / "weights/last.pt", best=run / "weights/best.pt", csv=run / "results.csv")
                for fn in self.callbacks["on_train_start"]: fn(trainer)
                for epoch in range(100):
                    trainer.epoch = epoch; state.epoch = epoch
                    trainer.last.write_bytes(b"last"); trainer.best.write_bytes(b"best")
                    trainer.csv.write_text("epoch,fitness\n" + "\n".join("%d,0" % n for n in range(1, epoch + 2)))
                    (run / "args.yaml").write_text("fixture")
                    (run / "sampler_audit.csv").write_text("epoch\n" + "\n".join(str(n) for n in range(1, epoch + 2)))
                    for fn in self.callbacks["on_model_save"]: fn(trainer)
        fake_torch = SimpleNamespace(__version__=config.EXPECTED_TORCH, version=SimpleNamespace(cuda="12.8"),
            cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 1, set_device=lambda _: None,
                                 get_device_name=lambda _: "Fake GPU", empty_cache=lambda: None),
            load=lambda *args, **kwargs: {"epoch": state.epoch, "optimizer": 1, "scaler": 1, "ema": 1, "updates": 1, "best_fitness": 1},
            save=lambda *args, **kwargs: None)
        fake_ultra = SimpleNamespace(__version__=config.EXPECTED_ULTRALYTICS, YOLO=YOLO)
        fake_hub = SimpleNamespace(HfApi=API, CommitOperationAdd=lambda **kwargs: SimpleNamespace(**kwargs), hf_hub_download=None)
        fake_yaml = SimpleNamespace(safe_load=json.loads, safe_dump=lambda value, **kwargs: json.dumps(value))
        with ExitStack() as stack:
            stack.enter_context(patch.dict(sys.modules, {"torch": fake_torch, "ultralytics": fake_ultra,
                "huggingface_hub": fake_hub, "yaml": fake_yaml}))
            stack.enter_context(patch.object(hn, "make_trainer", return_value=(object, lambda _: None, lambda _: None)))
            stack.enter_context(patch.object(hn.time, "sleep", return_value=None))
            original = Path.read_text
            def labels_only(path, *args, **kwargs):
                if path.parent == data / "labels/train" and path.suffix == ".txt": return ""
                return original(path, *args, **kwargs)
            stack.enter_context(patch.object(Path, "read_text", labels_only))
            stack.enter_context(redirect_stdout(io.StringIO()))
            hn.main(self.argv("--imgsz", "640", "--repo-id", "owner/repo"))
        self.assertEqual(len(state.train_calls), 1)
        call = state.train_calls[0]; self.assertEqual((call["epochs"], call["batch"], call["imgsz"], call["seed"]), (100, 35, 640, 43))
        self.assertEqual(len(state.commits), 12, "initialization, ten 10-epoch snapshots, then final upload")
        self.assertEqual(state.created["private"], True)
        self.assertEqual(Path.cwd(), start_cwd)
        remote = [op.path_in_repo for commit in state.commits for op in commit["operations"]]
        self.assertFalse(any(".env" in path or "HF_TOKEN" in path for path in remote))


class RecoveryTests(Fixture):
    """A real-file fake Hub snapshot: only SDK/ML boundaries are replaced."""
    def recovery_runtime(self, stack, fault=None):
        data = self.root / "data/grazpedwri_yolo"
        if not data.exists(): data = self.prepared_dataset()
        remote = self.root / "remote"; snapshot = remote / "epoch_090"; snapshot.mkdir(parents=True, exist_ok=True)
        (remote / "candidates.csv").write_bytes(self.candidate_bytes)
        histories = "epoch,fitness\n" + "\n".join("%d,0" % n for n in range(1, 91))
        audit = "epoch\n" + "\n".join(str(n) for n in range(1, 91))
        files = {"last.pt": b"checkpoint-90", "best.pt": b"best-90", "results.csv": histories.encode(),
                 "args.yaml": b"fixture", "sampler_audit.csv": audit.encode()}
        for name, body in files.items(): (snapshot / name).write_bytes(body)
        fingerprints = {name: hn.sha(path) for name, path in hn.source_files().items()}
        environment = {"imgsz": 640, "batch_size": 35, "seed": 43, "recipe": hn.RECIPE, "hard_weight": 3.0,
                       "candidate_csv_sha256": hashlib.sha256(self.candidate_bytes).hexdigest(), "source_sha256": fingerprints}
        (remote / "environment.json").write_text(json.dumps(environment))
        for name, source in hn.source_files().items(): (remote / name).write_bytes(Path(source).read_bytes())
        info = {"run_name": "old_run", "completed_epoch": 90, "stage_target_epochs": 100,
                "manifest_sha256": config.MANIFEST_SHA256, "hard_pool_sha256": config.HARD_NEGATIVE_POOL_SHA256,
                "files_sha256": {name: hn.sha(snapshot / name) for name in hn.SNAPSHOT_FILES}}
        (snapshot / "checkpoint_info.json").write_text(json.dumps(info))
        if fault == "snapshot": info["files_sha256"]["best.pt"] = "0" * 64; (snapshot / "checkpoint_info.json").write_text(json.dumps(info))
        if fault == "source": (remote / "config.py").write_text("corrupt captured source\n")
        checkpoint = {"epoch": 89, "optimizer": object(), "scaler": object(), "ema": object(), "updates": 1,
                      "best_fitness": 0.5, "train_args": {"epochs": 100, "imgsz": 640, "batch": 35, "seed": 43}}
        if fault == "state": checkpoint.pop("optimizer")
        if fault == "epochs": checkpoint["train_args"]["epochs"] = 50
        state = SimpleNamespace(downloads=[], commits=[], train_calls=[], epoch=99, data=data, checkpoint=checkpoint)
        class API:
            def whoami(self): return {"name": "fixture"}
            def create_repo(self, *args, **kwargs): return None
            def repo_info(self, *args, **kwargs): return SimpleNamespace(private=True)
            def file_exists(self, *args, **kwargs): return False
            def create_commit(self, **kwargs): state.commits.append(kwargs); return SimpleNamespace(commit_url="https://example.invalid/commit")
        class YOLO:
            def __init__(self, path): self.callbacks = defaultdict(list); self.ckpt_path = Path(path)
            def add_callback(self, name, fn): self.callbacks[name].append(fn)
            def train(self, **kwargs):
                state.train_calls.append(kwargs)
                run = Path(kwargs["save_dir"]); trainer = SimpleNamespace(save_dir=run, start_epoch=90, epochs=100,
                    batch_size=35, amp=True, epoch=99, last=run / "weights/last.pt", best=run / "weights/best.pt", csv=run / "results.csv")
                for fn in self.callbacks["on_train_start"]: fn(trainer)
                trainer.last.write_bytes(b"last-100"); trainer.best.write_bytes(b"best-100")
                trainer.csv.write_text("epoch,fitness\n" + "\n".join("%d,0" % n for n in range(1, 101)))
                (run / "args.yaml").write_text("fixture")
                (run / "sampler_audit.csv").write_text("epoch\n" + "\n".join(str(n) for n in range(1, 101)))
                for fn in self.callbacks["on_model_save"]: fn(trainer)
        def download(repo_id, filename, revision):
            state.downloads.append((repo_id, filename, revision))
            return str(remote / filename.removeprefix("runs/old_run/"))
        def load(path, **kwargs):
            return state.checkpoint if "epoch_090" in str(path) else {**state.checkpoint, "epoch": state.epoch,
                "optimizer": 1, "scaler": 1, "ema": 1, "train_args": state.checkpoint["train_args"]}
        torch = SimpleNamespace(__version__=config.EXPECTED_TORCH, version=SimpleNamespace(cuda="12.8"), load=load,
            save=lambda value, path: Path(path).write_bytes(b"resume"), cuda=SimpleNamespace(is_available=lambda: True,
            device_count=lambda: 1, set_device=lambda _: None, get_device_name=lambda _: "Fake GPU", empty_cache=lambda: None))
        hub = SimpleNamespace(HfApi=API, CommitOperationAdd=lambda **kw: SimpleNamespace(**kw), hf_hub_download=download)
        yaml = SimpleNamespace(safe_load=json.loads, safe_dump=lambda value, **kw: json.dumps(value))
        stack.enter_context(patch.dict(sys.modules, {"torch": torch, "ultralytics": SimpleNamespace(__version__=config.EXPECTED_ULTRALYTICS, YOLO=YOLO),
            "huggingface_hub": hub, "yaml": yaml}))
        stack.enter_context(patch.object(hn, "make_trainer", return_value=(object, lambda _: None, lambda _: None)))
        original = Path.read_text
        stack.enter_context(patch.object(Path, "read_text", lambda path, *a, **kw: "" if path.parent == data / "labels/train" and path.suffix == ".txt" else original(path, *a, **kw)))
        return state

    def recovery_args(self):
        return self.argv("--imgsz", "640", "--repo-id", "owner/repo", "--resume-hf-run", "old_run", "--resume-epoch", "90", "--hf-revision", "a" * 40)

    def test_epoch90_recovery_preserves_pinned_source_and_continues_to_100(self):
        before = Path.cwd()
        with ExitStack() as stack:
            state = self.recovery_runtime(stack)
            with redirect_stdout(io.StringIO()): hn.main(self.recovery_args())
        self.assertEqual(len(state.train_calls), 1)
        call = state.train_calls[0]
        self.assertEqual((call["resume"], call["batch"], call["device"]), (True, 35, 0))
        self.assertTrue(all(revision == "a" * 40 for _, _, revision in state.downloads))
        self.assertEqual({name for _, name, _ in state.downloads}, {"runs/old_run/environment.json", "runs/old_run/train_hard_negatives.py", "runs/old_run/config.py", "runs/old_run/candidates.csv", *["runs/old_run/epoch_090/" + f for f in (*hn.SNAPSHOT_FILES, "checkpoint_info.json")]})
        run = next(self.root.glob("runs/*")); env = json.loads((run / "environment.json").read_text())
        self.assertEqual((env["resume_source"], env["resume_commit"], env["stage_target_epochs"], env["completed_epochs"]), ("old_run", "a" * 40, 100, 100))
        self.assertEqual(env["resume_source_sha256"], {name: hn.sha(path) for name, path in hn.source_files().items()})
        self.assertFalse((self.root / "data/grazpedwri_yolo/images/test").exists())
        self.assertEqual(Path.cwd(), before)

    def test_recovery_rejects_bad_remote_artifacts_before_model_train(self):
        cases = (("snapshot", "Corrupted recovery file"), ("state", "Full-state recovery checkpoint required"),
                 ("epochs", "single 100-epoch schedule"), ("source", "Recovery source fingerprint differs"))
        for fault, message in cases:
            with self.subTest(fault=fault), ExitStack() as stack:
                state = self.recovery_runtime(stack, fault)
                with redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, message):
                    hn.main(self.recovery_args())
                self.assertEqual(state.train_calls, [], fault)
                self.assertTrue(all(revision == "a" * 40 for _, _, revision in state.downloads))


if __name__ == "__main__":
    unittest.main(verbosity=2)
