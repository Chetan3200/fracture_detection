"""Offline MedGemma configuration, launch and recovery-control regressions.

Synthetic local metadata only. No real training, images, GPU imports or Hub calls.
"""
from pathlib import Path
from unittest.mock import patch
import contextlib
import copy
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
MED = ROOT / "medgemma_training"
for directory in (ROOT, MED):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
import train_medgemma as training
import run_medgemma as launcher


class Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.env_patch = patch.dict(os.environ, {}, clear=True)
        self.env_patch.start()
        self.addCleanup(self.env_patch.stop)

    def plan(self, *extra):
        args, project = launcher.arguments(["--project-root", str(self.root), "--print-config", *extra])
        return args, project, launcher.make_plan(args, project)

    def smoke_fixture(self, *extra):
        args, project, plan = self.plan("train", "--devices", "2,3", "--repo-id", "test/private", *extra)
        inputs = {name: "f" * 64 for name in launcher.INPUT_FILES}
        inputs["split_manifest.csv"] = training.project_settings.MANIFEST_SHA256
        folder = project.runs_dir / f"medgemma_smoke_{plan['world_size']}gpu_20260921_000000_000000Z"
        folder.mkdir(parents=True)
        signature = {**launcher.expected_smoke_signature(plan), "input_hashes": inputs}
        run = {**launcher.current_hashes(), "signature": signature, "run_name": folder.name,
               "configuration": {"runtime_environment": plan["environment"]}}
        checkpoint = {"kind": "checkpoint", "repo_id": plan["repo_id"], "run_name": folder.name,
                      "world_size": plan["world_size"], "global_step": 4, "commit_sha": "a" * 40}
        final = {**checkpoint, "kind": "final", "commit_sha": "b" * 40}
        payloads = {"run_config.json": run, "training_summary.json": {"mode": "smoke", "global_step": 4},
                    "hf_backup_latest.json": final, "validation_format_summary.json": {"images": 4, "format_valid": 3}}
        for name, value in payloads.items():
            (folder / name).write_text(json.dumps(value))
        (folder / "hf_backup_receipts.jsonl").write_text("\n".join(json.dumps(r) for r in (checkpoint, final)) + "\n")
        return args, project, plan, folder, inputs, payloads, checkpoint, final

    def test_defaults_preserve_recipe_and_use_one_gpu(self):
        args, project, plan = self.plan()
        self.assertEqual(plan["environment"]["CUDA_VISIBLE_DEVICES"], "0")
        self.assertEqual(plan["world_size"], 1)
        self.assertEqual(plan["gradient_accumulation"], 16)
        self.assertEqual(plan["effective_batch"], 16)
        self.assertEqual(plan["training"], {"epochs": 3, "lr": 1e-4, "seed": 42, "lora_rank": 16,
            "attention": "sdpa", "max_seq_len": 2048, "generation_samples": 4, "max_new_tokens": 768, "backup_steps": 100})
        for item in ("torch.distributed.run", "--standalone", "--nnodes=1", "--nproc_per_node=1", "--smoke"):
            self.assertIn(item, plan["command"])
        self.assertEqual(plan["python"], str(self.root / ".venv/bin/python"))
        self.assertEqual(plan["data_root"], str(self.root / "data/grazpedwri_medgemma"))
        self.assertEqual(plan["environment"]["YOLO_AUTOINSTALL"], "false")
        self.assertNotIn("HF_HOME", plan["environment"])

    def test_full_command_roundtrips_through_trainer_parser(self):
        for mode, generation in (("smoke", 4), ("train", 16)):
            with self.subTest(mode=mode):
                _, _, plan = self.plan(mode, "--devices", "2,3", "--repo-id", "test/private")
                start = plan["command"].index(str(MED / "train_medgemma.py")) + 1
                args, project = training.resolve_arguments(plan["command"][start:])
                self.assertEqual(args.smoke, mode == "smoke")
                self.assertEqual(args.generation_samples, generation)
                self.assertEqual(args.grad_accum, 8)
                self.assertEqual(args.epochs, 3)
                self.assertEqual(args.backup_steps, 100)
                self.assertEqual(args.revision, training.project_settings.MEDGEMMA_BASE_REVISION)
                self.assertEqual(args.output, Path(plan["output"]))
                self.assertEqual(args.data_root, project.medgemma_data_dir)

    def test_arbitrary_gpu_counts_and_explicit_accumulation(self):
        for world in (1, 2, 3, 4, 5, 6, 7, 8, 16, 17, 32):
            for mode in ("smoke", "train"):
                with self.subTest(world=world, mode=mode):
                    accum = 16 // world if 16 % world == 0 else 3
                    extra = [] if 16 % world == 0 else ["--grad-accum", str(accum)]
                    _, _, plan = self.plan(mode, "--num-gpus", str(world), *extra)
                    self.assertEqual(plan["world_size"], world)
                    self.assertEqual(plan["gradient_accumulation"], accum)
                    self.assertEqual(plan["effective_batch"], world * accum)
                    self.assertEqual(plan["environment"]["CUDA_VISIBLE_DEVICES"], ",".join(map(str, range(world))))
                    self.assertIn(f"--nproc_per_node={world}", plan["command"])
                    self.assertIn(f"_{world}gpu_", Path(plan["output"]).name)
                    start = plan["command"].index(str(MED / "train_medgemma.py")) + 1
                    args, project = training.resolve_arguments(plan["command"][start:])
                    with patch.dict(os.environ, {"WORLD_SIZE": str(world)}):
                        resolved = training.resolved_configuration(args, project)
                    self.assertEqual(resolved["effective_batch"], world * accum)
                    self.assertEqual(resolved["gradient_accumulation"], accum)

    def test_nondivisors_do_not_silently_round_the_batch(self):
        for world in (3, 5, 6, 7, 17, 32):
            with self.subTest(world=world), self.assertRaisesRegex(ValueError, "--grad-accum explicitly"):
                self.plan("--num-gpus", str(world))
        _, _, plan = self.plan("--devices", "0,1,2", "--grad-accum", "4")
        self.assertEqual(plan["effective_batch"], 12)
        self.assertEqual(plan["world_size"], 3)
        for value in ("0", "-1"):
            with self.subTest(accum=value), self.assertRaisesRegex(ValueError, "accumulation"):
                self.plan("--num-gpus", "3", "--grad-accum=" + value)

    def test_count_override_respects_visible_devices_and_conflicts(self):
        for invalid in ("0", "-1"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                self.plan("--num-gpus=" + invalid)
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "3,5,7,9"}):
            _, _, plan = self.plan("--num-gpus", "2")
            self.assertEqual(plan["environment"]["CUDA_VISIBLE_DEVICES"], "3,5")
            self.assertEqual(plan["world_size"], 2)
            with self.assertRaisesRegex(ValueError, "exceeds"):
                self.plan("--num-gpus", "8")
            _, _, override = self.plan("--devices", "0,1,2", "--grad-accum", "4")
            self.assertEqual(override["environment"]["CUDA_VISIBLE_DEVICES"], "0,1,2")
        with self.assertRaisesRegex(ValueError, "must match"):
            self.plan("--devices", "0,1", "--num-gpus", "1")
        _, _, same = self.plan("--devices", "0,1,2", "--num-gpus", "3", "--grad-accum", "4")
        self.assertEqual(same["effective_batch"], 12)

    def test_accumulation_environment_precedence_and_direct_trainer(self):
        (self.root / ".env").write_text("MEDGEMMA_CUDA_VISIBLE_DEVICES=0,1,2\nMEDGEMMA_GRAD_ACCUM=4\n")
        _, _, plan = self.plan()
        self.assertEqual((plan["world_size"], plan["gradient_accumulation"], plan["effective_batch"]), (3, 4, 12))
        with patch.dict(os.environ, {"MEDGEMMA_GRAD_ACCUM": "5", "WORLD_SIZE": "3"}):
            _, _, from_env = self.plan()
            self.assertEqual(from_env["effective_batch"], 15)
            _, _, from_cli = self.plan("--grad-accum", "6")
            self.assertEqual(from_cli["effective_batch"], 18)
            args, project = training.resolve_arguments(["--project-root", str(self.root), "--print-config"])
            self.assertEqual(training.resolved_configuration(args, project)["effective_batch"], 15)
            args, project = training.resolve_arguments(["--project-root", str(self.root), "--print-config", "--grad-accum", "6"])
            self.assertEqual(training.resolved_configuration(args, project)["effective_batch"], 18)

    def test_smoke_gate_matches_each_gpu_count_and_batch(self):
        for world in (1, 3, 5, 8, 17):
            with self.subTest(world=world):
                devices = ",".join(map(str, range(world)))
                _, _, plan, folder, inputs, payloads, checkpoint, final = self.smoke_fixture(
                    "--devices", devices, "--grad-accum", "2")
                result = launcher.validate_smoke(folder, plan, launcher.current_hashes(), inputs)
                self.assertEqual(result["world_size"], world)
                for key, changed in (("world_size", world + 1), ("gradient_accumulation_steps", 3), ("effective_batch_size", 2 * world + 1)):
                    run = copy.deepcopy(payloads["run_config.json"])
                    run["signature"][key] = changed
                    (folder / "run_config.json").write_text(json.dumps(run))
                    with self.assertRaisesRegex(ValueError, "world size"):
                        launcher.validate_smoke(folder, plan, launcher.current_hashes(), inputs)
                (folder / "run_config.json").write_text(json.dumps(payloads["run_config.json"]))
                (folder / "hf_backup_receipts.jsonl").write_text("\n".join(json.dumps(r) for r in
                    ({**checkpoint, "world_size": world + 1}, final)))
                with self.assertRaisesRegex(ValueError, "checkpoint AND final"):
                    launcher.validate_smoke(folder, plan, launcher.current_hashes(), inputs)

    def test_cli_environment_dotenv_precedence_and_paths(self):
        (self.root / ".env").write_text("DATA_DIR=file-data\nRUNS_DIR=stored-runs\nHF_HOME=cache/hf\n"
            "MEDGEMMA_HF_REPO_ID=file/private\nMEDGEMMA_CUDA_VISIBLE_DEVICES=6,7\n"
            "NCCL_P2P_DISABLE=0\nHF_TOKEN=secret-not-for-output\n")
        _, _, plan = self.plan()
        self.assertEqual(plan["data_root"], str(self.root / "file-data/grazpedwri_medgemma"))
        self.assertEqual(plan["environment"]["CUDA_VISIBLE_DEVICES"], "6,7")
        self.assertEqual(plan["environment"]["HF_HOME"], str(self.root / "cache/hf"))
        self.assertEqual(plan["environment"]["NCCL_P2P_DISABLE"], "0")
        self.assertEqual(plan["repo_id"], "file/private")
        with patch.dict(os.environ, {"DATA_DIR": "env-data", "CUDA_VISIBLE_DEVICES": "4,5", "MEDGEMMA_HF_REPO_ID": "env/private"}):
            _, _, env_plan = self.plan()
            self.assertEqual(env_plan["environment"]["CUDA_VISIBLE_DEVICES"], "4,5")
            self.assertEqual(env_plan["repo_id"], "env/private")
            _, _, cli_plan = self.plan("--devices", "0,1", "--data-root", "cli-data", "--repo-id", "cli/private", "--output", "outputs/new")
            self.assertEqual(cli_plan["environment"]["CUDA_VISIBLE_DEVICES"], "0,1")
            self.assertEqual(cli_plan["data_root"], str(self.root / "cli-data"))
            self.assertEqual(cli_plan["repo_id"], "cli/private")
            self.assertEqual(cli_plan["output"], str(self.root / "outputs/new"))
        self.assertNotIn("secret-not-for-output", json.dumps(plan))
        self.assertNotIn("HF_TOKEN", os.environ)
        self.assertFalse((self.root / "cache").exists())

    def test_custom_env_file_and_executable(self):
        (self.root / "custom.env").write_text("MEDGEMMA_PYTHON=env/bin/python\nMEDGEMMA_HF_REPO_ID=custom/private\n")
        _, _, plan = self.plan("--env-file", "custom.env")
        self.assertEqual(plan["python"], str(self.root / "env/bin/python"))
        position = plan["command"].index("--env-file")
        self.assertEqual(plan["command"][position + 1], str(self.root / "custom.env"))
        _, _, override = self.plan("--env-file", "custom.env", "--python", "other/bin/python")
        self.assertEqual(override["python"], str(self.root / "other/bin/python"))

    def test_python_executable_does_not_dereference_venv_symlink(self):
        executable = self.root / ".venv/bin/python"
        executable.parent.mkdir(parents=True)
        executable.symlink_to(sys.executable)
        _, _, plan = self.plan()
        self.assertEqual(plan["python"], str(executable))
        self.assertNotEqual(plan["python"], str(executable.resolve()))

    def test_device_and_nccl_safeguards(self):
        for value in ("", "0,0", "00,0", "-1,0", "0,garbage", "0,"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "distinct CUDA"):
                self.plan("--devices=" + value)
        _, _, plan = self.plan("--devices", " 0, 1 ")
        self.assertEqual(plan["environment"]["CUDA_VISIBLE_DEVICES"], "0,1")
        with patch.dict(os.environ, {"NCCL_IB_DISABLE": "bad"}), self.assertRaisesRegex(ValueError, "NCCL"):
            self.plan()

    def test_unique_outputs_and_dataset_protection(self):
        _, _, one = self.plan()
        _, _, two = self.plan()
        self.assertNotEqual(one["output"], two["output"])
        self.assertFalse(Path(one["output"]).exists())
        with self.assertRaisesRegex(ValueError, "outside"):
            self.plan("--output", "data/grazpedwri_medgemma/bad")

    def test_missing_repo_can_preview_but_cannot_launch(self):
        self.plan()
        with patch.object(launcher.subprocess, "run", side_effect=AssertionError("must not launch")):
            with self.assertRaisesRegex(ValueError, "MEDGEMMA_HF_REPO_ID"):
                launcher.main(["--project-root", str(self.root)])
        with self.assertRaisesRegex(ValueError, "owner/repository"):
            self.plan("--repo-id", "https://example.com/token")

    def test_existing_output_blocks_launch(self):
        output = self.root / "existing"
        output.mkdir()
        with patch.object(launcher.subprocess, "run", side_effect=AssertionError("must not launch")):
            with self.assertRaisesRegex(ValueError, "Output already exists"):
                launcher.main(["--project-root", str(self.root), "--repo-id", "test/private",
                               "--python", sys.executable, "--output", str(output)])

    def test_successful_fake_smoke_and_formatting_not_accuracy_gate(self):
        _, _, plan, folder, inputs, payloads, _, _ = self.smoke_fixture()
        for valid in (0, 3, 4):
            (folder / "validation_format_summary.json").write_text(json.dumps({"images": 4, "format_valid": valid}))
            result = launcher.validate_smoke(folder, plan, launcher.current_hashes(), inputs)
            self.assertEqual(result["format_valid"], valid)
            self.assertEqual(result["world_size"], 2)

    def test_rejects_each_stale_source_hash(self):
        _, _, plan, folder, inputs, payloads, _, _ = self.smoke_fixture()
        for key in launcher.current_hashes():
            with self.subTest(key=key):
                run = copy.deepcopy(payloads["run_config.json"])
                run[key] = "0" * 64
                (folder / "run_config.json").write_text(json.dumps(run))
                with self.assertRaisesRegex(ValueError, "fingerprints"):
                    launcher.validate_smoke(folder, plan, launcher.current_hashes(), inputs)

    def test_rejects_wrong_world_recipe_revision_versions_and_data(self):
        _, _, plan, folder, inputs, payloads, _, _ = self.smoke_fixture()
        changes = {"world_size": 1, "mode": "full", "gradient_accumulation_steps": 4,
                   "base_revision": "0" * 40, "learning_rate": 0.01, "seed": 43,
                   "checkpoint_interval_steps": 0, "versions": {}, "input_hashes": {}}
        for key, value in changes.items():
            with self.subTest(key=key):
                run = copy.deepcopy(payloads["run_config.json"])
                run["signature"][key] = value
                (folder / "run_config.json").write_text(json.dumps(run))
                with self.assertRaises(ValueError):
                    launcher.validate_smoke(folder, plan, launcher.current_hashes(), inputs)

    def test_rejects_incomplete_training_generation_or_gpu_settings(self):
        _, _, plan, folder, inputs, payloads, _, _ = self.smoke_fixture()
        changes = [("training_summary.json", {"mode": "smoke", "global_step": 3}),
                   ("validation_format_summary.json", {"images": 3, "format_valid": 3}),
                   ("validation_format_summary.json", {"images": 4, "format_valid": 5})]
        for name, changed in changes:
            with self.subTest(name=name, changed=changed):
                (folder / name).write_text(json.dumps(changed))
                with self.assertRaises(ValueError):
                    launcher.validate_smoke(folder, plan, launcher.current_hashes(), inputs)
                (folder / name).write_text(json.dumps(payloads[name]))
        run = copy.deepcopy(payloads["run_config.json"])
        run["configuration"]["runtime_environment"]["CUDA_VISIBLE_DEVICES"] = "0,1"
        (folder / "run_config.json").write_text(json.dumps(run))
        with self.assertRaisesRegex(ValueError, "CUDA/NCCL"):
            launcher.validate_smoke(folder, plan, launcher.current_hashes(), inputs)

    def test_both_real_commit_receipts_to_matching_repository_required(self):
        _, _, plan, folder, inputs, _, checkpoint, final = self.smoke_fixture()
        broken = [[final], [checkpoint], [{**checkpoint, "world_size": 1}, final],
                  [{**checkpoint, "repo_id": "wrong/private"}, final],
                  [{**checkpoint, "commit_sha": ""}, final],
                  [checkpoint, {**final, "commit_sha": "c" * 40}]]
        for receipts in broken:
            with self.subTest(receipts=receipts):
                (folder / "hf_backup_receipts.jsonl").write_text("\n".join(json.dumps(r) for r in receipts))
                with self.assertRaisesRegex(ValueError, "checkpoint AND final"):
                    launcher.validate_smoke(folder, plan, launcher.current_hashes(), inputs)

    def test_find_smoke_skips_bad_newer_candidate_and_hashes_local_inputs(self):
        args, project, plan, folder, _, payloads, _, _ = self.smoke_fixture()
        data_root = Path(plan["data_root"])
        data_root.mkdir(parents=True)
        for name in launcher.INPUT_FILES:
            (data_root / name).write_bytes((ROOT / "split_manifest.csv").read_bytes() if name == "split_manifest.csv" else b"synthetic metadata\n")
        run = payloads["run_config.json"]
        run["signature"]["input_hashes"] = {name: training.file_digest(data_root / name) for name in launcher.INPUT_FILES}
        (folder / "run_config.json").write_text(json.dumps(run))
        bad = project.runs_dir / "medgemma_smoke_2gpu_9999"
        bad.mkdir()
        (bad / "run_config.json").write_text("{bad json")
        result = launcher.find_smoke(args, project, plan)
        self.assertEqual(result["smoke_run"], str(folder))
        args.smoke_run = bad
        with self.assertRaisesRegex(ValueError, "Selected smoke"):
            launcher.find_smoke(args, project, plan)
        (data_root / "split_manifest.csv").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "frozen manifest"):
            launcher.find_smoke(args, project, plan)

    def test_check_ready_exits_without_starting_a_process(self):
        with patch.object(launcher, "find_smoke", return_value={"verified": "fixture only"}), \
             patch.object(launcher.subprocess, "run", side_effect=AssertionError("no launch")), \
             contextlib.redirect_stdout(io.StringIO()):
            result = launcher.main(["train", "--project-root", str(self.root), "--repo-id", "test/private", "--check-ready"])
            self.assertEqual(result, 0)

    def test_child_launch_uses_exact_plan_environment_and_return_code_with_fake_process(self):
        (self.root / ".env").write_text("HF_TOKEN=private-sentinel\nHF_HOME=hf-cache\n")
        output = self.root / "new-run"
        with patch.object(launcher.subprocess, "run") as child, contextlib.redirect_stdout(io.StringIO()) as text:
            child.return_value.returncode = 7
            result = launcher.main(["smoke", "--project-root", str(self.root), "--repo-id", "test/private",
                                    "--python", sys.executable, "--output", str(output)])
        self.assertEqual(result, 7)
        self.assertEqual(child.call_count, 1)
        call = child.call_args
        self.assertEqual(call.kwargs["cwd"], str(self.root))
        self.assertEqual(call.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "0")
        self.assertEqual(call.kwargs["env"]["HF_HOME"], str(self.root / "hf-cache"))
        self.assertEqual(call.kwargs["env"]["HF_TOKEN"], "private-sentinel")
        self.assertNotIn("private-sentinel", text.getvalue())
        self.assertFalse(output.exists())

    def test_trainer_defaults_paths_and_revision_on_resume(self):
        args, project = training.resolve_arguments(["--project-root", str(self.root), "--print-config"])
        self.assertEqual(args.data_root, self.root / "data/grazpedwri_medgemma")
        self.assertIsNone(args.output)
        self.assertEqual(args.revision, training.project_settings.MEDGEMMA_BASE_REVISION)
        args, _ = training.resolve_arguments(["--project-root", str(self.root), "--output", "runs/original",
                                              "--resume", "runs/original/checkpoint-100", "--print-config"])
        self.assertIsNone(args.revision)  # Actual recovery reuses the SAVED revision.
        self.assertEqual(args.resume.parent, args.output)
        with self.assertRaisesRegex(ValueError, "requires --output"):
            training.arguments(["--project-root", str(self.root)])

    def test_invalid_trainer_parameters_rejected_offline(self):
        for extra in (("--lr", "nan"), ("--lr", "0"), ("--epochs", "0"), ("--grad-accum", "0"),
                      ("--seed", "-1"), ("--backup-steps", "-1"), ("--max-new-tokens", "0"), ("--revision", "main")):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                training.arguments(["--project-root", str(self.root), "--print-config", *extra])

    def test_supported_world_and_device_guards(self):
        for world in (1, 2, 3, 4, 5, 6, 8, 16, 17, 32):
            for rank in range(world):
                training.validate_rank_configuration(world, rank, rank, world)
            if 16 % world == 0:
                self.assertEqual(training.default_accumulation(world) * world, 16)
        for values in ((0, 0, 0, 3), (1, 0, 0, 2), (2, 2, 0, 2), (2, 0, -1, 2), (2, 1, 2, 2), (1, 0, 0, 0)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                training.validate_rank_configuration(*values)
        with patch.dict(os.environ, {"WORLD_SIZE": "2"}):
            args, project = training.resolve_arguments(["--project-root", str(self.root), "--print-config"])
            self.assertEqual(training.resolved_configuration(args, project)["gradient_accumulation"], 8)

    def test_resume_checks_actual_script_helper_and_config_sources(self):
        files = [self.root / name for name in ("training_script.py", "medgemma_support.py", "config.py")]
        for filename in files:
            filename.write_text("# verified fixture\n")
        keys = ("script_sha256", "helper_sha256", "config_sha256")
        previous = dict(zip(keys, (training.file_digest(filename) for filename in files)))
        training.validate_resume_sources(previous, *files)
        for key in keys:
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "Resume"):
                training.validate_resume_sources({**previous, key: "0" * 64}, *files)
        training.validate_resume_sources({"script_sha256": previous["script_sha256"]}, *files)
        files[0].write_text("# altered trainer\n")
        with self.assertRaisesRegex(ValueError, "trainer source differs"):
            training.validate_resume_sources(previous, *files)

    def test_previews_do_not_activate_environment_train_or_spawn(self):
        (self.root / ".env").write_text("HF_TOKEN=do-not-leak\nHF_HOME=hf-cache\n")
        before = sorted(str(p) for p in self.root.rglob("*"))
        with patch.object(training, "run_training", side_effect=AssertionError("no training")), \
             patch.object(launcher.subprocess, "run", side_effect=AssertionError("no child")), \
             contextlib.redirect_stdout(io.StringIO()) as text:
            training.main(["--project-root", str(self.root), "--print-config"])
            launcher.main(["--project-root", str(self.root), "--print-config"])
        self.assertNotIn("do-not-leak", text.getvalue())
        self.assertNotIn("HF_TOKEN", os.environ)
        self.assertEqual(before, sorted(str(p) for p in self.root.rglob("*")))

    def test_standalone_archive_and_previews_block_gpu_imports_network_and_nested_launch(self):
        archived = self.root / "restored"
        archived.mkdir()
        shutil.copy2(MED / "train_medgemma.py", archived / "training_script.py")
        shutil.copy2(MED / "medgemma_support.py", archived / "medgemma_support.py")
        shutil.copy2(ROOT / "config.py", archived / "config.py")
        guard = '''import builtins, runpy, socket, subprocess, sys
original = builtins.__import__
blocked = {"torch", "transformers", "peft", "bitsandbytes", "huggingface_hub", "requests", "kagglehub"}
def guarded(name, *a, **kw):
    if name.split(".")[0] in blocked: raise AssertionError("GPU/network dependency imported: " + name)
    return original(name, *a, **kw)
def deny(*a, **kw): raise AssertionError("Network or nested subprocess attempted")
builtins.__import__ = guarded
socket.socket = deny
subprocess.run = deny
script, root, *flags = sys.argv[1:]
sys.argv = [script, "--project-root", root, *flags]
runpy.run_path(script, run_name="__main__")
'''
        for script in (MED / "train_medgemma.py", MED / "run_medgemma.py", archived / "training_script.py"):
            variants = [["--help"], ["--print-config"]]
            if script == MED / "run_medgemma.py":
                variants += [["--num-gpus", "3", "--grad-accum", "4", "--print-config"],
                             ["--num-gpus", "8", "--print-config"],
                             ["--num-gpus", "32", "--grad-accum", "1", "--print-config"]]
            for flags in variants:
                with self.subTest(script=script, flags=flags):
                    # The bundled test interpreter needs its stdlib prefix when
                    # the surrounding process environment is intentionally empty.
                    completed = subprocess.run([sys.executable, "-B", "-S", "-c", guard, str(script), str(self.root), *flags],
                        cwd=str(self.root), env={"PYTHONDONTWRITEBYTECODE": "1", "PYTHONHOME": sys.base_prefix},
                        capture_output=True, text=True)
                    self.assertEqual(completed.returncode, 0, completed.stderr)
                    if "--print-config" in flags:
                        json.loads(completed.stdout)
        self.assertFalse((self.root / "runs").exists())
        self.assertFalse((self.root / "data").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
