"""Portable, standard-library configuration for the frozen prediction backends.

The evaluation engine and prediction helpers remain byte-stable: their SHA256
values are embedded in completed validation protocols. These frontends only
resolve paths/options and launch those files in the existing environment.
Preview reads configuration/protocol/source metadata, never weights or images.
"""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import math
import os
import re
import shlex
import subprocess

import config

SOURCE_ROOT = Path(__file__).resolve().parent


def require(condition, message):
    if not condition:
        raise ValueError(message)


def lexical_path(value, root):
    """Keep venv and .pt cache symlink names; resolving them can change behavior."""
    result = Path(value).expanduser()
    return Path(os.path.abspath(result if result.is_absolute() else root / result))


def source_files(mode, model):
    evaluation = SOURCE_ROOT / "evaluation_scripts"
    folder = evaluation if mode == "evaluate" else SOURCE_ROOT / "inference_scripts"
    entry = folder / f"{'evaluate' if mode == 'evaluate' else 'infer'}_{model}.py"
    common = folder / ("evaluation_common.py" if mode == "evaluate" else "inference_common.py")
    files = [entry, common, evaluation / "checkpoints.py"]
    if model == "medgemma":
        files.append(evaluation / "medgemma_model.py")
    return files


def resolve_arguments(mode, argv=None):
    require(mode in {"evaluate", "infer"}, "Unknown prediction workflow.")
    parser = argparse.ArgumentParser(description=f"Configured {mode} frontend. --print-config is offline; real execution uses the existing .venv.")
    parser.add_argument("model", choices=("yolo26", "medgemma"))
    parser.add_argument("--project-root", type=Path, help="Data/config workspace; defaults to this checkout")
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--python", type=Path, help="Existing executable; default PREDICTION_PYTHON or PROJECT_ROOT/.venv/bin/python")
    parser.add_argument("--print-config", action="store_true", help="Preview without ML imports, downloads, image/weight reads or output creation")
    checkpoint = parser.add_mutually_exclusive_group()
    checkpoint.add_argument("--checkpoint", type=Path)
    checkpoint.add_argument("--hf", action="store_true")
    parser.add_argument("--hf-repo", "--repo-id", dest="hf_repo", help="HF_REPO_ID for YOLO; MEDGEMMA_HF_REPO_ID for MedGemma")
    parser.add_argument("--hf-revision", help="Optional exact 40-character HF commit; otherwise backend resolves HEAD once")
    parser.add_argument("--hf-filename", help="YOLO only: exact .pt path inside the model repository")
    parser.add_argument("--hf-run", help="MedGemma only: completed run name inside the model repository")
    parser.add_argument("--cache-dir", type=Path, help="Default CACHE_DIR/fracture_evaluation")
    parser.add_argument("--output", type=Path, help="NEW directory; default unique name inside RUNS_DIR")
    parser.add_argument("--device", type=int, help="One logical CUDA index; default PREDICTION_DEVICE or 0")
    parser.add_argument("--imgsz", type=int, help="YOLO only; saved protocol setting or 640")
    parser.add_argument("--batch", type=int, help="YOLO only; saved protocol setting or 8")
    parser.add_argument("--max-new-tokens", type=int, help="MedGemma only; saved protocol/training setting")
    if mode == "evaluate":
        parser.add_argument("--dataset", type=Path, help="BOTH models use DATA_DIR/grazpedwri_yolo")
        parser.add_argument("--split", choices=("val", "test"), default="val")
        parser.add_argument("--protocol", type=Path, help="Required for test: matching COMPLETE validation evaluation.json")
        for name in ("bootstraps", "bootstrap_seed", "benchmark_runs"):
            parser.add_argument("--" + name.replace("_", "-"), type=int, default=config.EVALUATION_DEFAULTS[name])
        parser.add_argument("--diagnostics", action="store_true")
        parser.add_argument("--smoke", action="store_true", help="Validation-only four-image check, not a usable test protocol")
    else:
        parser.add_argument("--source", type=Path, help="One image or directory, not DICOM or a glob")
        parser.add_argument("--recursive", action="store_true")
        cutoff = parser.add_mutually_exclusive_group()
        cutoff.add_argument("--protocol", type=Path, help="Matching COMPLETE validation evaluation.json; inherits cutoff and prediction settings")
        cutoff.add_argument("--conf", type=float, help="Explicit exploratory cutoff, NOT calibrated confidence")
        parser.add_argument("--no-annotate", action="store_true")
    args = parser.parse_args(argv)
    project = config.load_config(args.project_root, args.env_file)
    args.mode = mode
    args.device = args.device if args.device is not None else int(project.setting("PREDICTION_DEVICE", "0"))
    require(args.device >= 0, "CUDA device must be nonnegative.")
    if args.model == "medgemma":
        require(args.imgsz is None and args.batch is None and args.hf_filename is None, "--imgsz/--batch/--hf-filename are YOLO-only.")
    else:
        require(args.max_new_tokens is None and args.hf_run is None, "--max-new-tokens/--hf-run are MedGemma-only.")
    require(args.hf or all(getattr(args, key) is None for key in ("hf_repo", "hf_revision", "hf_filename", "hf_run")),
            "Use --hf with HF options; do not combine them with local checkpoints.")
    if args.hf:
        args.hf_repo = args.hf_repo or (project.hf_repo_id if args.model == "yolo26" else project.medgemma_hf_repo_id)
        if args.hf_repo:
            require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*", args.hf_repo) is not None,
                    "HF repository must be owner/repository, not a URL or credential.")
    if args.hf_revision is not None:
        require(re.fullmatch(r"[a-fA-F0-9]{40}", args.hf_revision) is not None, "--hf-revision must be an immutable 40-character commit.")
        args.hf_revision = args.hf_revision.lower()
    if mode == "evaluate":
        require((args.split == "test") == (args.protocol is not None), "Only test requires --protocol from completed validation.")
        require(not args.smoke or args.split == "val", "Smoke cannot use the held-out test split.")
        require(args.bootstraps >= 100 and args.benchmark_runs >= 10, "Use >=100 bootstraps and >=10 benchmark runs.")
        require(args.bootstrap_seed >= 0, "Bootstrap seed must be nonnegative.")
        args.dataset = config.resolve_path(args.dataset or project.yolo_data_dir, project.project_root)
    elif args.conf is not None:
        require(math.isfinite(args.conf) and 0.001 <= args.conf <= 1, "--conf must be finite and between 0.001 and 1; filtering is score > cutoff.")
    for key in ("checkpoint", "protocol", "source"):
        value = getattr(args, key, None)
        if value is not None:
            setattr(args, key, lexical_path(value, project.project_root))
    args.cache_dir = config.resolve_path(args.cache_dir or project.cache_dir / "fracture_evaluation", project.project_root)
    args.python = lexical_path(args.python or project.setting("PREDICTION_PYTHON", str(project.project_root / ".venv/bin/python")), project.project_root)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%fZ")
    label = f"evaluation_{args.model}_{args.split}" if mode == "evaluate" else f"inference_{args.model}"
    args.output = config.resolve_path(args.output or project.runs_dir / f"{label}_{stamp}", project.project_root)
    if mode == "evaluate":
        require(not args.output.is_relative_to(args.dataset), "Put evaluation output outside the prepared dataset.")
    elif args.source is not None and args.source.is_dir():
        require(not args.output.is_relative_to(args.source.resolve()), "Put inference output outside the source directory.")
    require(not args.cache_dir.is_relative_to(args.output), "Put cache outside the new output directory.")
    return args, project


def read_protocol(args):
    if args.protocol is None:
        return None
    require(args.protocol.is_file() and args.protocol.stat().st_size <= 16 * 1024**2, "Missing or oversized validation protocol.")
    report = json.loads(args.protocol.read_text(encoding="utf-8"))
    require(isinstance(report, dict) and report.get("status") == "complete" and report.get("split") == "val",
            "Use a COMPLETE validation evaluation.json, not a test, smoke or legacy report.")
    protocol = report.get("protocol")
    require(isinstance(protocol, dict) and protocol.get("selected_on") == "val", "Protocol was not selected on validation.")
    identity = protocol.get("identity")
    require(isinstance(identity, dict) and identity.get("model") == args.model, "Validation protocol belongs to a different model.")
    require(identity.get("raw_manifest_sha256") == config.MANIFEST_SHA256, "Protocol uses a different frozen manifest.")
    cutoff = protocol.get("confidence_cutoff")
    require(type(cutoff) in (int, float) and math.isfinite(cutoff) and 0.001 <= cutoff <= 1, "Invalid validation cutoff.")
    require(identity.get("score_floor") == 0.001 and identity.get("max_detections") == 300
            and identity.get("confidence_comparison") == "score > cutoff", "Unsupported validation collection/filtering rules.")
    # Early, CPU-only source check. Backends still verify the COMPLETE applicable
    # checkpoint/data/software/prediction identity before using this protocol.
    files = source_files(args.mode, args.model)
    checked = files if args.mode == "evaluate" else files[2:]
    expected = {filename.name: config.sha256(filename) for filename in checked}
    saved = identity.get("source_sha256", {})
    require(isinstance(saved, dict) and all(saved.get(name) == digest for name, digest in expected.items())
            and (args.mode != "evaluate" or set(saved) == set(expected)),
            "Source differs from this validation protocol. Use its matching implementation; do not edit the report or bypass checks.")
    return report


def protocol_setting(report, explicit, keys, default):
    if report is None:
        return default if explicit is None else explicit
    value = report["protocol"]["identity"].get("inference")
    for key in keys:
        require(isinstance(value, dict) and key in value, "Protocol lacks setting: " + ".".join(keys))
        value = value[key]
    require(explicit is None or explicit == value, "Explicit " + ".".join(keys) + " differs from validation; omit it to reuse the saved value.")
    return value


def make_plan(args, project):
    report = read_protocol(args)
    if args.model == "yolo26":
        args.imgsz = protocol_setting(report, args.imgsz, ("collection", "imgsz"), config.EVALUATION_DEFAULTS["imgsz"])
        args.batch = protocol_setting(report, args.batch, ("prediction_batch",), config.EVALUATION_DEFAULTS["batch"])
        require(type(args.imgsz) is int and args.imgsz > 0 and args.imgsz % 32 == 0
                and type(args.batch) is int and args.batch > 0, "Invalid YOLO image size/batch.")
    else:
        args.max_new_tokens = protocol_setting(report, args.max_new_tokens, ("max_new_tokens",), None)
        require(args.max_new_tokens is None or (type(args.max_new_tokens) is int and args.max_new_tokens > 0), "Invalid generation token limit.")
    missing = []
    if not args.hf and args.checkpoint is None:
        missing.append("--checkpoint or --hf")
    if args.hf and not args.hf_repo:
        missing.append("--hf-repo or configured model repository")
    if args.hf and args.model == "yolo26" and not args.hf_filename:
        missing.append("--hf-filename")
    if args.mode == "infer":
        if args.source is None:
            missing.append("--source")
        if args.protocol is None and args.conf is None:
            missing.append("--protocol or explicit exploratory --conf")
    files = source_files(args.mode, args.model)
    command = [str(args.python), "-u", str(files[0])]
    if args.hf:
        command.append("--hf")
    for key in ("checkpoint", "hf_repo", "hf_revision", "hf_filename", "hf_run", "cache_dir", "output", "device", "protocol"):
        value = getattr(args, key)
        if value is not None:
            command += ["--" + key.replace("_", "-"), str(value)]
    if args.model == "yolo26":
        command += ["--imgsz", str(args.imgsz), "--batch", str(args.batch)]
    elif args.max_new_tokens is not None:
        command += ["--max-new-tokens", str(args.max_new_tokens)]
    if args.mode == "evaluate":
        for key in ("dataset", "split", "bootstraps", "bootstrap_seed", "benchmark_runs"):
            command += ["--" + key.replace("_", "-"), str(getattr(args, key))]
        for key in ("diagnostics", "smoke"):
            if getattr(args, key):
                command.append("--" + key)
    else:
        for key in ("source", "conf"):
            if getattr(args, key) is not None:
                command += ["--" + key, str(getattr(args, key))]
        for key in ("recursive", "no_annotate"):
            if getattr(args, key):
                command.append("--" + key.replace("_", "-"))
    environment = {key: project.setting(key) for key in ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER") if project.setting(key) is not None}
    environment.update(YOLO_AUTOINSTALL="false", PYTHONUNBUFFERED="1")
    if project.hf_home is not None:
        environment["HF_HOME"] = str(project.hf_home)
    return {"workflow": args.mode, "model": args.model, "project": project.public_dict(), "source_root": str(SOURCE_ROOT),
            "python": str(args.python), "output": str(args.output), "cache_dir": str(args.cache_dir), "device": args.device,
            "dataset": str(args.dataset) if args.mode == "evaluate" else None,
            "source": str(args.source) if args.mode == "infer" and args.source is not None else None,
            "split": args.split if args.mode == "evaluate" else None,
            "checkpoint": str(args.checkpoint) if args.checkpoint is not None else None,
            "hf_repo": args.hf_repo, "hf_revision": args.hf_revision,
            "prediction": {"imgsz": args.imgsz, "batch": args.batch, "max_new_tokens": args.max_new_tokens},
            "protocol": str(args.protocol) if args.protocol is not None else None,
            "protocol_sha256": config.sha256(args.protocol) if args.protocol is not None else None,
            "confidence_cutoff": report["protocol"]["confidence_cutoff"] if report else getattr(args, "conf", None),
            "protocol_runtime_identity": report["protocol"]["identity"] if report else None,
            "backend_source_sha256": {filename.name: config.sha256(filename) for filename in files},
            "launcher_source_sha256": {filename.name: config.sha256(filename) for filename in
                                       (SOURCE_ROOT / f"{args.mode}.py", Path(__file__), Path(config.__file__))},
            "environment": environment, "command": command, "missing_for_execution": missing,
            "verification": "Preview only checks configuration/protocol/source metadata; backend checks actual checkpoint, data and software at execution."}


def main(mode, argv=None):
    args, project = resolve_arguments(mode, argv)
    plan = make_plan(args, project)
    if args.print_config:
        print(json.dumps(plan, indent=2, allow_nan=False))
        return 0
    require(not plan["missing_for_execution"], "Missing: " + ", ".join(plan["missing_for_execution"]))
    require(project.project_root.is_dir(), "Configured project workspace does not exist.")
    require(args.python.is_file() and os.access(args.python, os.X_OK), "Existing Python executable not found; use the project .venv or --python. No packages were installed.")
    require(not os.path.lexists(args.output), "Output already exists; choose a NEW directory.")
    project.activate_environment()
    environment = os.environ.copy()  # Auth may be inherited by the child; NEVER serialize or print this mapping.
    environment.update(plan["environment"])
    print("Run long jobs inside tmux. Command:", shlex.join(plan["command"]), flush=True)
    result = subprocess.run(plan["command"], cwd=str(project.project_root), env=environment, check=False)
    # Backends own output creation and report contents. Never rewrite their reports,
    # pre-create their output directory, or write into a failed/racing launch.
    if result.returncode == 0 and args.output.is_dir():
        receipt = args.output / "launcher_config.json"
        with receipt.open("x", encoding="utf-8") as handle:
            json.dump({**plan, "exit_code": 0}, handle, indent=2, allow_nan=False)
            handle.write("\n")
    return result.returncode
