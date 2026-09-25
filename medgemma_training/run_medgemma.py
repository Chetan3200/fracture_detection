"""Configurable single-node MedGemma smoke/full launcher. Run long jobs inside tmux.

Uses the existing project .venv; never installs packages or changes Torch/CUDA.
--print-config only previews a command. --check-ready only checks local smoke
reports. Neither mode imports GPU libraries, downloads models or contacts HF.
Full training requires a matching completed smoke AND checkpoint/final receipts.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
import os
import re
import shlex
import subprocess
import sys

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import train_medgemma as training

config = training.project_settings
require = training.require
INPUT_FILES = ("split_manifest.csv", "prompt.txt", "train.jsonl", "val.jsonl", "val_inputs.jsonl",
               "dataset_info.json", "processor_requirements.json")


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", nargs="?", choices=("smoke", "train"), default="smoke")
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output", type=Path, help="Optional NEW run directory; otherwise generate a unique name in RUNS_DIR")
    parser.add_argument("--repo-id", "--hf-repo-id", dest="repo_id", help="Private destination; default MEDGEMMA_HF_REPO_ID")
    parser.add_argument("--python", type=Path, help="Existing environment executable; default PROJECT_ROOT/.venv/bin/python")
    parser.add_argument("--devices", help="Comma-separated CUDA device IDs; count inferred. Default configured visible devices or GPU 0")
    parser.add_argument("--num-gpus", type=int, help="Use N GPUs: first N configured visible devices, otherwise IDs 0..N-1")
    parser.add_argument("--grad-accum", type=int, help="Accumulation per GPU; MEDGEMMA_GRAD_ACCUM or exact 16/N when possible")
    parser.add_argument("--smoke-run", type=Path, help="Check a specific completed smoke directory instead of scanning RUNS_DIR")
    preview = parser.add_mutually_exclusive_group()
    preview.add_argument("--print-config", action="store_true", help="Print non-secret launch plan and exit, even without data/.venv")
    preview.add_argument("--check-ready", action="store_true", help="For train mode: validate local smoke evidence and exit; no launch")
    args = parser.parse_args(argv)
    require(not args.check_ready or args.mode == "train", "--check-ready is for train mode.")
    if args.num_gpus is not None:
        training.validate_world_size(args.num_gpus)
    project = config.load_config(args.project_root, args.env_file)
    args.repo_id = args.repo_id.strip() if args.repo_id is not None else project.medgemma_hf_repo_id
    if args.repo_id:
        require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*", args.repo_id) is not None,
                "HF repository must be owner/repository, not a URL or credential.")
    require(args.repo_id or args.print_config, "Set MEDGEMMA_HF_REPO_ID or --repo-id; the smoke gate requires both private uploads.")
    return args, project


def make_plan(args, project):
    if args.devices is not None:
        devices = args.devices
    else:
        # Process-level masks outrank .env, even when a different alias was used.
        devices = next((os.environ[k] for k in ("MEDGEMMA_CUDA_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES") if k in os.environ),
            project.setting("MEDGEMMA_CUDA_VISIBLE_DEVICES", project.setting("CUDA_VISIBLE_DEVICES")))
    if devices is None:
        devices = (",".join(str(index) for index in range(args.num_gpus)) if args.num_gpus is not None
                   else config.MEDGEMMA_GPU_DEFAULTS["CUDA_VISIBLE_DEVICES"])
    tokens = [value.strip() for value in devices.split(",")]
    tokens = [str(int(value)) if value.isdigit() else value for value in tokens]
    require(tokens and len(set(tokens)) == len(tokens) and all(
        re.fullmatch(r"(?:[0-9]+|GPU-[A-Za-z0-9-]+|MIG-[A-Za-z0-9_./-]+)", value) for value in tokens),
        "Select one or more distinct CUDA device IDs with --devices or MEDGEMMA_CUDA_VISIBLE_DEVICES.")
    if args.num_gpus is not None:
        if args.devices is not None:
            require(len(tokens) == args.num_gpus, "--num-gpus must match the explicit --devices list.")
        else:
            require(args.num_gpus <= len(tokens), "--num-gpus exceeds the configured visible-device list; select devices explicitly.")
            tokens = tokens[:args.num_gpus]
    world = len(tokens)
    training.validate_world_size(world)
    configured_accum = project.setting("MEDGEMMA_GRAD_ACCUM")
    accum = args.grad_accum
    if accum is None and configured_accum is not None and str(configured_accum).strip():
        accum = int(configured_accum)
    accum = training.default_accumulation(world) if accum is None else accum
    require(accum > 0, "Gradient accumulation must be positive.")
    environment = {key: str(project.setting(key, value)) for key, value in config.MEDGEMMA_GPU_DEFAULTS.items()
                   if key != "CUDA_VISIBLE_DEVICES"}
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(tokens)
    require(environment["CUDA_DEVICE_ORDER"] in {"PCI_BUS_ID", "FASTEST_FIRST"}, "Invalid CUDA_DEVICE_ORDER.")
    require(all(environment[key] in {"0", "1"} for key in ("NCCL_P2P_DISABLE", "NCCL_IB_DISABLE")), "NCCL disable flags must be 0 or 1.")
    environment.update(YOLO_AUTOINSTALL="false", PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false", HF_HUB_DISABLE_TELEMETRY="1")
    if project.hf_home is not None:
        environment["HF_HOME"] = str(project.hf_home)
    # Do NOT resolve the executable symlink: doing so can escape a Python venv.
    executable = Path(args.python or project.setting("MEDGEMMA_PYTHON", str(project.project_root / ".venv/bin/python"))).expanduser()
    if not executable.is_absolute():
        executable = project.project_root / executable
    executable = Path(os.path.abspath(executable))
    data_root = config.resolve_path(args.data_root or project.medgemma_data_dir, project.project_root)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%fZ")
    output = config.resolve_path(args.output or project.runs_dir / f"medgemma_{args.mode}_{world}gpu_{stamp}", project.project_root)
    require(not output.is_relative_to(data_root), "Put training output outside the prepared dataset.")
    command = [str(executable), "-u", "-m", "torch.distributed.run", "--standalone", "--nnodes=1", f"--nproc_per_node={world}",
               str(HERE / "train_medgemma.py"), "--project-root", str(project.project_root), "--data-root", str(data_root),
               "--output", str(output), "--revision", config.MEDGEMMA_BASE_REVISION, "--grad-accum", str(accum)]
    if args.env_file is not None:
        command += ["--env-file", str(project.env_file)]
    if args.repo_id:
        command += ["--hf-repo-id", args.repo_id]
    if args.mode == "smoke":
        command.append("--smoke")
    recipe = dict(config.MEDGEMMA_DEFAULTS)
    recipe["generation_samples"] = 4 if args.mode == "smoke" else config.MEDGEMMA_DEFAULTS["generation_samples"]
    for key, value in recipe.items():
        command += ["--" + key.replace("_", "-"), str(value)]
    return {"mode": args.mode, "project_root": str(project.project_root), "data_root": str(data_root),
            "output": str(output), "python": str(executable), "repo_id": args.repo_id, "world_size": world,
            "gradient_accumulation": accum, "effective_batch": world * accum, "base_revision": config.MEDGEMMA_BASE_REVISION,
            "training": recipe, "environment": environment, "command": command,
            "smoke_gate": "Checked before full launch; preview is not evidence of GPU or upload readiness"}


def current_hashes():
    return {"script_sha256": training.file_digest(HERE / "train_medgemma.py"),
            "helper_sha256": training.file_digest(HERE / "medgemma_support.py"),
            "config_sha256": training.file_digest(config.__file__)}


def expected_smoke_signature(plan):
    recipe = config.MEDGEMMA_DEFAULTS
    return {"model_id": training.MODEL_ID, "base_revision": plan["base_revision"], "mode": "smoke", "world_size": plan["world_size"],
            "microbatch_per_gpu": 1, "gradient_accumulation_steps": plan["gradient_accumulation"], "effective_batch_size": plan["effective_batch"],
            "epochs": recipe["epochs"], "max_steps": 4, "learning_rate": recipe["lr"], "seed": recipe["seed"],
            "checkpoint_interval_steps": recipe["backup_steps"], "lora_rank": recipe["lora_rank"],
            "lora_alpha": 2 * recipe["lora_rank"], "lora_dropout": 0.05, "attention": recipe["attention"],
            "max_seq_len_guard": recipe["max_seq_len"], "max_new_tokens": recipe["max_new_tokens"],
            "versions": {**training.VERSIONS, "torch": config.EXPECTED_TORCH}}


def validate_smoke(folder, plan, hashes, input_hashes):
    folder = Path(folder)
    run = json.loads((folder / "run_config.json").read_text())
    summary = json.loads((folder / "training_summary.json").read_text())
    latest = json.loads((folder / "hf_backup_latest.json").read_text())
    formatting = json.loads((folder / "validation_format_summary.json").read_text())
    receipts = [json.loads(line) for line in (folder / "hf_backup_receipts.jsonl").read_text().splitlines() if line.strip()]
    require(all(run.get(key) == value for key, value in hashes.items()), "Smoke source fingerprints are stale or missing.")
    signature = run["signature"]
    require(all(signature.get(key) == value for key, value in expected_smoke_signature(plan).items()),
            "Smoke recipe/base revision/versions/world size differ.")
    require(signature.get("input_hashes") == input_hashes, "Smoke data/prompt hashes differ from the configured dataset.")
    require(run.get("run_name") == folder.name, "Smoke run name does not match its directory.")
    runtime = run.get("configuration", {}).get("runtime_environment", {})
    require(all(runtime.get(key) == plan["environment"][key] for key in config.MEDGEMMA_GPU_DEFAULTS),
            "Smoke CUDA/NCCL settings differ from this launch.")
    require(summary.get("global_step") == 4 and summary.get("mode") == "smoke", "Smoke training did not complete four updates.")
    require(formatting.get("images") == 4 and type(formatting.get("format_valid")) is int
            and 0 <= formatting["format_valid"] <= 4, "Smoke formatting check did not cover four images.")

    def uploaded(receipt, kind):
        return (receipt.get("kind") == kind and receipt.get("world_size") == plan["world_size"] and receipt.get("global_step") == 4
                and receipt.get("repo_id") == plan["repo_id"] and receipt.get("run_name") == folder.name
                and isinstance(receipt.get("commit_sha"), str)
                and re.fullmatch(r"[a-f0-9]{40}", receipt["commit_sha"]) is not None)
    require(uploaded(latest, "final") and any(uploaded(receipt, "checkpoint") for receipt in receipts)
            and any(uploaded(receipt, "final") and receipt["commit_sha"] == latest["commit_sha"] for receipt in receipts),
            "Smoke needs successful checkpoint AND final uploads to this private target repo.")
    return {"smoke_run": str(folder), "world_size": plan["world_size"], "global_step": 4,
            "format_valid": formatting["format_valid"], "format_images": 4, "final_commit": latest["commit_sha"],
            "note": "Local receipt-based readiness only; formatting is not localization accuracy."}


def find_smoke(args, project, plan):
    data_root = Path(plan["data_root"])
    inputs = {name: training.file_digest(data_root / name) for name in INPUT_FILES}
    require(inputs["split_manifest.csv"] == config.MANIFEST_SHA256, "Configured dataset has the wrong frozen manifest.")
    hashes = current_hashes()
    folders = ([config.resolve_path(args.smoke_run, project.project_root)] if args.smoke_run is not None
               else sorted(project.runs_dir.glob(f"medgemma_smoke_{plan['world_size']}gpu_*"), reverse=True))
    for folder in folders:
        try:
            return validate_smoke(folder, plan, hashes, inputs)
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            if args.smoke_run is not None:
                raise ValueError(f"Selected smoke is not ready: {exc}") from exc
    raise ValueError(f"No matching completed {plan['world_size']}-GPU smoke with both uploads. Run smoke with the same devices and accumulation first.")


def main(argv=None):
    args, project = arguments(argv)
    plan = make_plan(args, project)
    if args.print_config:
        print(json.dumps(plan, indent=2, allow_nan=False))
        return 0
    if args.mode == "train":
        readiness = find_smoke(args, project, plan)
        print(json.dumps(readiness, indent=2))
        if args.check_ready:
            return 0
    require(project.project_root.is_dir(), "Configured project workspace does not exist.")
    require(Path(plan["python"]).is_file() and os.access(plan["python"], os.X_OK),
            "Existing Python environment executable not found. Use the project .venv or pass --python; no packages were installed.")
    require(not os.path.lexists(plan["output"]), "Output already exists; choose a new run directory.")
    project.activate_environment()
    environment = os.environ.copy()  # Passed to the child only; NEVER print or serialize this mapping.
    environment.update(plan["environment"])
    print("Run inside tmux. Command:", shlex.join(plan["command"]), flush=True)
    print("Output:", plan["output"], flush=True)
    result = subprocess.run(plan["command"], cwd=str(project.project_root), env=environment, check=False)
    print(f"MedGemma {args.mode} exit code: {result.returncode} (0 = success)", flush=True)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
