"""Train a fresh YOLO26s baseline with one continuous schedule (default: 100 epochs).

Examples from the repository root, using the existing .venv:
  python train_baseline.py --imgsz 640 --seed 43 --print-config
  CUDA_VISIBLE_DEVICES=0 python train_baseline.py --imgsz 640 --seed 43 --repo-id owner/repo
  CUDA_VISIBLE_DEVICES=2 python train_baseline.py --imgsz 960 --seed 43 --no-hf-backup

Defaults: 640px/batch 35 or 960px/batch 14; private HF backups every 10 epochs.
No GPU libraries, downloads or HF calls are used by --help or --print-config.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
import os
import re
import shutil
import sys
import time

import config

CODE_ROOT = Path(__file__).resolve().parent


def require(ok, message):
    if not ok:
        raise ValueError(message)


def resolve_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project-root", type=Path, help="Data/config workspace root; defaults to this repository. Source snapshots always use the executed checkout.")
    parser.add_argument("--env-file", type=Path, help="Default: .env in the project root; explicit files must exist")
    parser.add_argument("--dataset", type=Path, help="Prepared YOLO directory; default DATA_DIR/grazpedwri_yolo")
    parser.add_argument("--manifest", type=Path, help="Frozen root manifest; default MANIFEST_PATH")
    parser.add_argument("--output-root", type=Path, help="Parent of a NEW timestamped run; default RUNS_DIR")
    parser.add_argument("--imgsz", type=int, choices=tuple(config.YOLO_BATCH_SIZES), default=640)
    parser.add_argument("--batch", type=int, help="Default: 35 at 640px; 14 at 960px; must remain fixed")
    parser.add_argument("--epochs", type=int, default=config.YOLO_EPOCHS)
    parser.add_argument("--seed", type=int, default=config.YOLO_SEED)
    parser.add_argument("--device", type=int, default=0, help="Logical GPU index after CUDA_VISIBLE_DEVICES")
    parser.add_argument("--backup-every", type=int, default=config.BACKUP_EVERY)
    parser.add_argument("--repo-id", help="Private HF destination; overrides HF_REPO_ID")
    parser.add_argument("--no-hf-backup", action="store_true", help="Local-only training; no HF login/uploads required")
    parser.add_argument("--print-config", action="store_true", help="Print safe resolved settings and exit; no data/GPU/network work")
    args = parser.parse_args(argv)
    project = config.load_config(args.project_root, args.env_file)
    args.dataset = config.resolve_path(args.dataset or project.yolo_data_dir, project.project_root)
    args.manifest = config.resolve_path(args.manifest or project.manifest_path, project.project_root)
    args.output_root = config.resolve_path(args.output_root or project.runs_dir, project.project_root)
    require(not args.output_root.is_relative_to(args.dataset), "Put output runs outside the prepared dataset directory.")
    args.batch = config.YOLO_BATCH_SIZES[args.imgsz] if args.batch is None else args.batch
    args.repo_id = args.repo_id.strip() if args.repo_id is not None else project.hf_repo_id
    require(args.epochs > 0 and args.batch > 0 and args.device >= 0, "Epochs/batch must be positive and device nonnegative.")
    require(0 <= args.seed < (1 << 32) - 1, "Seed must be between 0 and 2**32 - 2.")
    require(args.backup_every > 0, "Backup interval must be positive.")
    if not args.no_hf_backup:
        require(args.epochs % args.backup_every == 0, "Epochs must end on an HF backup boundary.")
        if args.repo_id:
            require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*", args.repo_id) is not None,
                    "--repo-id/HF_REPO_ID must be owner/repository, not a URL or credential.")
        require(args.repo_id or args.print_config,
                "Set HF_REPO_ID in .env/environment, pass --repo-id, or explicitly select --no-hf-backup.")
    return args, project


def training_settings(args):
    """Explicit values matching the existing pinned YOLO training recipe."""
    return {
        "imgsz": args.imgsz, "epochs": args.epochs, "batch": args.batch, "patience": 0,
        "optimizer": "SGD", "lr0": 0.01, "lrf": 0.01, "momentum": 0.937,
        "weight_decay": 0.0005, "nbs": 64, "device": args.device, "amp": True,
        "seed": args.seed, "deterministic": True, "workers": 2, "cache": False,
        "val": True, "split": "val", "nms": None, "rect": False,
        "cos_lr": False, "close_mosaic": 10, "save": True, "save_period": -1, "plots": True,
    }


def resolved_configuration(args, project):
    # Deliberately do not serialize argparse argv, os.environ or raw .env contents.
    return {
        "schema_version": 1,
        "recipe": f"COCO; single {args.epochs}-epoch schedule; ordinary baseline sampling",
        "project": project.public_dict(),
        "dataset": str(args.dataset), "manifest": str(args.manifest), "output_root": str(args.output_root),
        "manifest_sha256": config.MANIFEST_SHA256,
        "pretrained_model": "yolo26s.pt", "training": training_settings(args),
        "hf_backup": {"enabled": not args.no_hf_backup,
                      "repo_id": args.repo_id if not args.no_hf_backup else None,
                      "every_epochs": args.backup_every if not args.no_hf_backup else None},
        "expected_versions": {"torch": config.EXPECTED_TORCH, "ultralytics": config.EXPECTED_ULTRALYTICS},
    }


def source_files():
    # Snapshot the code actually executed, even when data lives in another workspace.
    return {"train_baseline.py": Path(__file__).resolve(),
            "config.py": Path(config.__file__).resolve(),
            "data_prep/prepare_data.py": CODE_ROOT / "data_prep/prepare_data.py"}


def validate_dataset(args):
    """Read local metadata before any GPU/model/Hub work. Never read test labels/images."""
    import yaml
    for path in [args.dataset / "data.yaml", args.dataset / "dataset_info.json",
                 args.dataset / "split_manifest.csv", args.manifest, *source_files().values()]:
        require(path.is_file(), f"Required file missing: {path}")
    info = json.loads((args.dataset / "dataset_info.json").read_text())
    require(info.get("manifest_sha256") == config.MANIFEST_SHA256, "Dataset metadata has a different frozen split.")
    for path in [args.manifest, args.dataset / "split_manifest.csv"]:
        actual = config.sha256(path)
        require(actual == config.MANIFEST_SHA256,
                f"Manifest checksum mismatch at {path}: expected {config.MANIFEST_SHA256}, found {actual}")
    data = yaml.safe_load((args.dataset / "data.yaml").read_text())
    require(isinstance(data, dict), "data.yaml must contain a mapping.")
    for split in ("train", "val"):
        require(data.get(split) == f"images/{split}", f"Unexpected {split} split in data.yaml.")
        for directory in (args.dataset / "images" / split, args.dataset / "labels" / split):
            require(directory.is_dir(), f"Prepared dataset directory missing: {directory}")
    names = data.get("names")
    require((isinstance(names, list) and names == ["fracture"]) or
            (isinstance(names, dict) and names == {0: "fracture"}), "Expected one class: 0 = fracture.")
    # Correct moved-server paths ONLY in the new run; do not edit the shared dataset.
    data["path"] = str(args.dataset)
    return info, data


def main(argv=None):
    args, project = resolve_arguments(argv)
    settings = resolved_configuration(args, project)
    if args.print_config:
        print(json.dumps(settings, indent=2, allow_nan=False))
        return
    info, data = validate_dataset(args)
    project.activate_environment()
    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

    # Lazy imports keep help/config inspection independent of the GPU environment.
    import torch
    import ultralytics
    import yaml
    from ultralytics import YOLO
    require(ultralytics.__version__ == config.EXPECTED_ULTRALYTICS,
            f"Use the pinned environment with ultralytics=={config.EXPECTED_ULTRALYTICS}; no packages were installed.")
    require(str(torch.__version__) == config.EXPECTED_TORCH,
            f"Use the existing pinned torch=={config.EXPECTED_TORCH} environment.")
    require(torch.cuda.is_available() and args.device < torch.cuda.device_count(), "Requested CUDA device unavailable.")
    torch.cuda.set_device(args.device)
    torch.ones(1, device=f"cuda:{args.device}").add_(1)
    torch.cuda.synchronize(args.device)

    api, add_operation, hub_version = None, None, None
    if not args.no_hf_backup:
        import huggingface_hub
        from huggingface_hub import CommitOperationAdd, HfApi
        add_operation, hub_version = CommitOperationAdd, huggingface_hub.__version__
        api = HfApi()
        api.whoami()
        api.create_repo(repo_id=args.repo_id, repo_type="model", private=True, exist_ok=True)
        require(api.repo_info(repo_id=args.repo_id, repo_type="model").private is True,
                "Refusing to upload checkpoints to a public repository.")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%fZ")
    run_name = f"baseline_yolo26s_{args.imgsz}_{args.epochs}epochs_seed{args.seed}_{stamp}"
    run_dir = args.output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
    for name in ("split_manifest.csv", "dataset_info.json"):
        shutil.copy2(args.dataset / name, run_dir / name)
    snapshots = []
    for name, original in source_files().items():
        destination = run_dir / "source" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, destination)
        require(config.sha256(destination) == config.sha256(original), f"Source copy mismatch: {name}")
        snapshots.append(destination)
    settings["source_sha256"] = {p.relative_to(run_dir / "source").as_posix(): config.sha256(p) for p in snapshots}
    settings["run_name"] = run_name
    settings["run_dir"] = str(run_dir)
    (run_dir / "resolved_config.json").write_text(json.dumps(settings, indent=2, allow_nan=False))

    old_cwd = Path.cwd()
    model = None
    try:
        os.chdir(project.project_root)
        model = YOLO("yolo26s.pt")  # Fresh COCO initialization, never an experiment checkpoint.
        environment = {
            "python": sys.version, "python_executable": sys.executable,
            "ultralytics": ultralytics.__version__, "torch": str(torch.__version__),
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(args.device), "device": args.device,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "manifest_sha256": info["manifest_sha256"], "recipe": settings["recipe"],
            "pretrained_model": "yolo26s.pt", "pretrained_weights_sha256": config.sha256(model.ckpt_path),
            "validation_head": "one-to-many with NMS (nms=None)",
            "epochs": args.epochs, "image_size": args.imgsz, "batch_size": args.batch, "seed": args.seed,
            "hf_repository": args.repo_id if api is not None else None,
            "hf_backup_every_epochs": args.backup_every if api is not None else None,
            "huggingface_hub": hub_version, "source_sha256": settings["source_sha256"],
        }
        (run_dir / "environment.json").write_text(json.dumps(environment, indent=2, allow_nan=False))
        common_files = [run_dir / n for n in ("environment.json", "resolved_config.json", "split_manifest.csv",
                                             "dataset_info.json", "data.yaml")] + snapshots

        def upload(files, message):
            final_error = None
            for attempt in range(1, config.UPLOAD_RETRIES + 1):
                try:
                    require(api.repo_info(repo_id=args.repo_id, repo_type="model").private is True,
                            "Refusing to upload to a public repository.")
                    # Rebuild operations on every attempt; HF may mutate upload objects.
                    return api.create_commit(repo_id=args.repo_id, repo_type="model", commit_message=message,
                        operations=[add_operation(path_in_repo=remote, path_or_fileobj=str(local))
                                    for remote, local in files])
                except Exception as exc:
                    final_error = exc
                    print(f"HF backup attempt {attempt}/{config.UPLOAD_RETRIES} failed: {type(exc).__name__}", flush=True)
                    if attempt < config.UPLOAD_RETRIES:
                        time.sleep((15, 60)[attempt - 1])
            raise RuntimeError("HF backup failed; training stopped and local files remain.") from final_error

        def common_uploads():
            return [(f"runs/{run_name}/{p.relative_to(run_dir).as_posix()}", p) for p in common_files]

        if api is not None:
            upload(common_uploads(), f"{run_name}: initialize checkpoint backup")
        uploaded_epochs = set()

        def verify_batch(trainer):
            require(trainer.batch_size == args.batch, "Training batch changed; refusing silent OOM batch reduction.")

        def verify_start(trainer):
            require(Path(trainer.save_dir).resolve() == run_dir and trainer.start_epoch == 0 and trainer.epochs == args.epochs,
                    "Unexpected output directory or training schedule.")
            verify_batch(trainer)
            print(f"VERIFIED: epochs 1-{args.epochs}, {args.imgsz}px, batch {args.batch}, seed {args.seed}", flush=True)

        def upload_checkpoint(trainer):
            epoch = int(trainer.epoch) + 1
            verify_batch(trainer)
            if api is None or epoch % args.backup_every or epoch in uploaded_epochs:
                return
            weights = Path(trainer.save_dir) / "weights"
            last, best = weights / "last.pt", weights / "best.pt"
            require(last.is_file() and last.stat().st_size > 0, f"Missing last.pt after epoch {epoch}")
            require(best.is_file() and best.stat().st_size > 0, f"Missing best.pt after epoch {epoch}")
            snapshot = run_dir / f"hf_backup_epoch_{epoch:03d}.json"
            snapshot.write_text(json.dumps({
                "run_name": run_name, "completed_epoch": epoch, "total_epochs": args.epochs,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "last_pt_sha256": config.sha256(last), "best_pt_sha256": config.sha256(best),
                "manifest_sha256": config.MANIFEST_SHA256, "repo_id": args.repo_id,
                "remote_folder": f"runs/{run_name}/epoch_{epoch:03d}",
            }, indent=2, allow_nan=False))
            prefix = f"runs/{run_name}/epoch_{epoch:03d}"
            files = [(f"{prefix}/last.pt", last), (f"{prefix}/best.pt", best),
                     (f"{prefix}/checkpoint_info.json", snapshot)] + common_uploads()
            for name in ("args.yaml", "results.csv"):
                file = Path(trainer.save_dir) / name
                if file.is_file():
                    files.append((f"{prefix}/{name}", file))
            commit = upload(files, f"{run_name}: backup after epoch {epoch}/{args.epochs}")
            uploaded_epochs.add(epoch)
            print(f"HF BACKUP COMPLETE: epoch {epoch} -> {commit.commit_url}", flush=True)

        model.add_callback("on_train_start", verify_start)
        model.add_callback("on_train_epoch_start", verify_batch)
        # on_model_save runs after complete checkpoint writes and before final stripping.
        model.add_callback("on_model_save", upload_checkpoint)
        print(f"Run folder: {run_dir}\nHF backups: {args.repo_id if api is not None else 'disabled (local only)'}", flush=True)
        model.train(data=str(run_dir / "data.yaml"), project=str(args.output_root), name=run_name,
                    exist_ok=True, **training_settings(args))
        require(int(model.trainer.epoch) + 1 == args.epochs,
                "Training returned before all requested epochs completed; local checkpoints remain.")
        if api is not None:
            require(args.epochs in uploaded_epochs, "Final HF backup did not complete.")
        print(f"\nRun folder: {run_dir}\nBest checkpoint: {run_dir / 'weights/best.pt'}", flush=True)
        print(f"Latest checkpoint: {run_dir / 'weights/last.pt'}", flush=True)
        if api is not None:
            print(f"Remote backups: https://huggingface.co/{args.repo_id}/tree/main/runs/{run_name}")
        print("The test split has not been evaluated.")
    finally:
        os.chdir(old_cwd)


if __name__ == "__main__":
    main()
