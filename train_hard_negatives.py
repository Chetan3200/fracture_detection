#!/usr/bin/env python3
"""YOLO26s HN experiment: COCO -> one continuous 100-epoch schedule.

One flat baseline-style run folder; one rolling recovery/ folder; private HF
snapshots every 10 epochs. The sampler changes PRIMARY image draws only:
9,479 positives once + 3,961 weighted negative draws with replacement.
Blank review fields mean annotation-defined approval, NOT clinical confirmation.
Defaults use the tracked ranked_negatives.csv and its frozen checksum.
CLI > exported environment > .env > shared config. --help / --print-config
are offline and need no Torch, CUDA or HF packages. Expose one GPU for training.
"""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import argparse
import csv
import gc
import hashlib
import io
import json
import math
import os
import random
import re
import shutil
import sys
import time

import config

MANIFEST_SHA = config.MANIFEST_SHA256
BATCH = config.YOLO_BATCH_SIZES
N_POS = config.SPLIT_COUNTS["train"]["positive"]
N_NEG = config.SPLIT_COUNTS["train"]["negative"]
HARD_WEIGHT = config.HARD_NEGATIVE_WEIGHT
EPOCHS, BACKUP_EVERY = config.YOLO_EPOCHS, config.BACKUP_EVERY
RECIPE = "COCO; single 100-epoch schedule; primary HN sampling"
SNAPSHOT_FILES = ("last.pt", "best.pt", "results.csv", "args.yaml", "sampler_audit.csv")


def check(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def rows(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def check_history(path, epoch):
    data = rows(path)
    key = next((k for k in (data[0] if data else {}) if k.strip() == "epoch"), None)
    check(key and [int(float(r[key])) for r in data] == list(range(1, epoch + 1)),
          f"Incomplete epoch history: {path}")


def load_pool(candidate_bytes, dataset=None, manifest=None):
    """Validate the frozen cohort and either supported representation of its HN CSV.

    CLI callers supply resolved paths. Defaults use the shared project config.
    Only TRAIN negative labels are read; validation/test labels are never used.
    """
    if dataset is None or manifest is None:
        project = config.load_config()
        dataset = project.yolo_data_dir if dataset is None else dataset
        manifest = project.manifest_path if manifest is None else manifest
    data = config.resolve_path(dataset)
    manifest = config.resolve_path(manifest)
    manifest_bytes = (data / "split_manifest.csv").read_bytes()
    check(hashlib.sha256(manifest_bytes).hexdigest() == MANIFEST_SHA, "Wrong frozen dataset split.")
    check(sha(manifest) == MANIFEST_SHA, "Configured root manifest differs.")
    check(json.loads((data / "dataset_info.json").read_text())["manifest_sha256"] == MANIFEST_SHA,
          "Dataset provenance differs from the frozen split.")
    cohort, patient_splits = {}, {}
    for r in csv.DictReader(io.StringIO(manifest_bytes.decode("utf-8-sig"))):
        pid, split = int(r["patient_id"]), r["split"]
        check(patient_splits.setdefault(pid, split) == split, "Patient leakage across splits.")
        if split == "train" and r["sample_type"] in {"positive", "negative"}:
            stem, positive = r["filestem"], r["sample_type"] == "positive"
            check(stem not in cohort and (int(r["fracture_count"]) > 0) == positive, "Invalid training cohort.")
            if not positive:
                check(json.loads(r["fracture_labels_json"]) == [] and
                      not (data / "labels/train" / f"{stem}.txt").read_text().strip(), f"Invalid negative: {stem}")
            cohort[stem] = (pid, positive)
    check(Counter(v[1] for v in cohort.values()) == {True: N_POS, False: N_NEG}, "Training counts changed.")
    selected, seen = [], set()
    for r in csv.DictReader(io.StringIO(candidate_bytes.decode("utf-8-sig"))):
        stem = r["image_id"]
        check(stem not in seen, "Duplicate candidate-table image.")
        seen.add(stem)
        if "candidate" in r:
            proposed = r["candidate"].strip().lower() == "yes"
            decision = r.get("review", "").strip().lower()
        else:
            check("candidate_status" in r, "Unrecognized mining CSV format.")
            proposed = r["candidate_status"] == "candidate_pending_review"
            decision = r.get("review_decision", "").strip().lower()
        if not proposed:
            continue
        check(decision in {"", "keep", "exclude", "uncertain"}, "Unknown review decision.")
        if decision in {"exclude", "uncertain"}:
            continue
        check(stem in cohort and not cohort[stem][1] and int(r["patient_id"]) == cohort[stem][0],
              f"Not an audited TRAIN negative: {stem}")
        score = float(r["hardness_score"])
        check(math.isfinite(score) and 0.10 <= score <= 1, "Candidate violates the mining floor.")
        selected.append({"image_id": stem, "patient_id": cohort[stem][0], "hardness_score": f"{score:.6f}",
                         "review": decision or "annotation_defined"})
    check(0 < len(selected) <= int(0.20 * N_NEG), "Empty or oversized hard-negative pool.")
    check(max(Counter(r["patient_id"] for r in selected).values()) <= 2, "Pool exceeds two images per patient.")
    text = io.StringIO(newline="")
    writer = csv.DictWriter(text, ["image_id", "patient_id", "hardness_score", "review"])
    writer.writeheader()
    writer.writerows(sorted(selected, key=lambda r: r["image_id"]))
    pool_bytes = text.getvalue().encode("utf-8")
    check(len(selected) == config.HARD_NEGATIVE_COUNT, "The frozen HN pool must contain exactly 178 images.")
    check(hashlib.sha256(pool_bytes).hexdigest() == config.HARD_NEGATIVE_POOL_SHA256,
          "Normalized hard-negative pool differs from the frozen experiment.")
    return cohort, selected, pool_bytes


class PrimarySampler:
    """Fixed epoch length, independent sampling RNG, identical ID order at both resolutions."""
    def __init__(self, ids, cohort, hard, seed):
        self.ids, self.seed, self.plan = list(ids), seed, None
        check(len(set(ids)) == len(ids) and set(ids) == set(cohort), "Loader/manifest cohort mismatch.")
        ordered = sorted(range(len(ids)), key=lambda i: ids[i])
        self.pos = [i for i in ordered if cohort[ids[i]][1]]
        self.neg = [i for i in ordered if not cohort[ids[i]][1]]
        self.weights = [HARD_WEIGHT if ids[i] in hard else 1.0 for i in self.neg]

    def set_epoch(self, epoch):
        rng = random.Random(self.seed + epoch)
        self.plan = self.pos + rng.choices(self.neg, weights=self.weights, k=len(self.neg))
        rng.shuffle(self.plan)

    def __len__(self):
        return len(self.ids)

    def __iter__(self):
        check(self.plan is not None, "Set sampler epoch before iteration.")
        return iter(self.plan)


def make_trainer(cohort, hard, seed, batch, audit):
    import torch
    from torch.utils.data import DataLoader
    from ultralytics.models.yolo.detect import DetectionTrainer
    from ultralytics.data.build import seed_worker

    class EpochLoader(DataLoader):
        def __iter__(self):
            # Lazy creation avoids starting a discarded iterator before mosaic reset.
            yield from super().__iter__()

        def reset(self):
            iterator = getattr(self, "_iterator", None)
            if iterator is not None and hasattr(iterator, "_shutdown_workers"):
                iterator._shutdown_workers()
            self._iterator = None

        close = reset

    class HNTrainer(DetectionTrainer):
        def get_dataloader(self, path, batch_size=16, rank=0, mode="train"):
            if mode != "train":
                return super().get_dataloader(path, batch_size, rank, mode)
            check(rank == -1 and batch_size == batch, "One GPU and the matched batch are required; no OOM batch reduction.")
            dataset = self.build_dataset(path, mode, batch_size)
            ids = [Path(p).stem for p in dataset.im_files]
            sampler = PrimarySampler(ids, cohort, hard, seed)
            check(not dataset.rect, "Rectangular training would change the baseline.")
            for stem, label in zip(ids, dataset.labels):
                check((len(label["cls"]) > 0) == cohort[stem][1], "Loader labels disagree with manifest.")
            workers = min(os.cpu_count() or 1, self.args.workers)
            generator = torch.Generator().manual_seed((6148914691236517204 + torch.initial_seed()) % (1 << 64))
            return EpochLoader(dataset, batch_size=batch, sampler=sampler, num_workers=workers,
                               persistent_workers=workers > 0, prefetch_factor=4 if workers else None,
                               pin_memory=True, collate_fn=dataset.collate_fn, worker_init_fn=seed_worker,
                               generator=generator, drop_last=False)

        def preprocess_batch(self, batch_data):
            self.hn_seen.extend(Path(p).stem for p in batch_data["im_file"])
            return super().preprocess_batch(batch_data)

    def begin(t):
        check(t.batch_size == batch, "Training batch changed.")
        t.train_loader.sampler.set_epoch(t.epoch)
        t.hn_seen = []

    def finish(t):
        sampler = t.train_loader.sampler
        expected = [sampler.ids[i] for i in sampler.plan]
        check(t.hn_seen == expected, "Consumed samples differ from the planned weighted epoch.")
        counts = Counter(t.hn_seen)
        check(all(counts[sampler.ids[i]] == 1 for i in sampler.pos), "A positive was skipped or repeated.")
        record = {"epoch": t.epoch + 1, "positive_draws": len(sampler.pos), "negative_draws": len(sampler.neg),
                  "hard_negative_draws": sum(counts[s] for s in hard),
                  "sequence_sha256": hashlib.sha256("\n".join(expected).encode()).hexdigest()}
        # Native NaN recovery can retry an epoch: replace its audit, do not duplicate it.
        previous = [r for r in rows(audit) if int(r["epoch"]) < record["epoch"]] if audit.exists() else []
        with audit.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, record.keys())
            writer.writeheader()
            writer.writerows(previous + [record])
        print(f"\nHN epoch {t.epoch + 1}: {N_POS} positives + {N_NEG} negatives; {record['hard_negative_draws']} hard draws.", flush=True)
    return HNTrainer, begin, finish


def resolve_arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-root", type=Path, help="Data/config workspace; defaults to this repository")
    p.add_argument("--env-file", type=Path)
    p.add_argument("--dataset", type=Path, help="Prepared YOLO directory; default DATA_DIR/grazpedwri_yolo")
    p.add_argument("--manifest", "--manifest-path", dest="manifest", type=Path, help="Default MANIFEST_PATH")
    p.add_argument("--output-root", type=Path, help="Parent of a NEW timestamped run; default RUNS_DIR")
    p.add_argument("--imgsz", type=int, choices=tuple(BATCH), required=True)
    p.add_argument("--candidates", type=Path, help="Default CANDIDATES_PATH (tracked ranked_negatives.csv)")
    p.add_argument("--candidate-sha256", help="Default frozen ranked CSV hash; alternate formats must select the SAME frozen pool")
    p.add_argument("--seed", type=int, default=config.YOLO_SEED)
    p.add_argument("--repo-id", help="Private model repository; default HF_REPO_ID")
    p.add_argument("--resume-hf-run", help="Recover an interrupted HN run into a new folder, preserving its 100-epoch schedule")
    p.add_argument("--resume-epoch", type=int)
    p.add_argument("--hf-revision", help="Exact HF commit containing the recovery checkpoint")
    p.add_argument("--print-config", action="store_true", help="Print safe resolved settings; no dataset/GPU/network work")
    args = p.parse_args(argv)
    project = config.load_config(args.project_root, args.env_file)
    args.dataset = config.resolve_path(args.dataset or project.yolo_data_dir, project.project_root)
    args.manifest = config.resolve_path(args.manifest or project.manifest_path, project.project_root)
    args.output_root = config.resolve_path(args.output_root or project.runs_dir, project.project_root)
    check(not args.output_root.is_relative_to(args.dataset), "Put output runs outside the prepared dataset.")
    check(EPOCHS == 100 and BACKUP_EVERY == 10, "This frozen HN recipe requires a continuous 100-epoch schedule and 10-epoch backups.")
    check(0 <= args.seed < (1 << 32) - 1, "Invalid seed.")
    args.repo_id = args.repo_id.strip() if args.repo_id is not None else project.hf_repo_id
    if args.repo_id:
        check(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*", args.repo_id) is not None,
              "HF repository must be owner/repository, not a URL or credential.")
    check(args.repo_id or args.print_config, "Set HF_REPO_ID in .env/environment or pass --repo-id for private backups.")
    if args.resume_hf_run:
        check(re.fullmatch(r"[A-Za-z0-9_.-]+", args.resume_hf_run) is not None and args.resume_hf_run not in {".", ".."},
              "Invalid source run name.")
        check(args.resume_epoch in range(BACKUP_EVERY, EPOCHS, BACKUP_EVERY) and args.hf_revision and
              re.fullmatch(r"[0-9a-fA-F]{40}", args.hf_revision), "Recovery requires a prior 10-epoch boundary and pinned HF commit.")
        check(args.candidates is None and args.candidate_sha256 is None, "Recovery uses its already-frozen pool.")
        args.hf_revision = args.hf_revision.lower()
    else:
        check(args.resume_epoch is None and args.hf_revision is None, "Incomplete recovery options.")
        args.candidates = config.resolve_path(args.candidates or project.candidates_path, project.project_root)
        args.candidate_sha256 = args.candidate_sha256 or config.CANDIDATE_CSV_SHA256
        check(re.fullmatch(r"[0-9a-fA-F]{64}", args.candidate_sha256) is not None, "Invalid candidate CSV SHA256.")
        args.candidate_sha256 = args.candidate_sha256.lower()
    return args, project


def resolved_configuration(args, project):
    return {
        "schema_version": 1, "recipe": RECIPE, "project": project.public_dict(),
        "dataset": str(args.dataset), "manifest": str(args.manifest), "output_root": str(args.output_root),
        "candidates": str(args.candidates) if args.candidates is not None else None,
        "candidate_csv_sha256": args.candidate_sha256, "manifest_sha256": MANIFEST_SHA,
        "hard_pool_sha256": config.HARD_NEGATIVE_POOL_SHA256,
        "epochs": EPOCHS, "imgsz": args.imgsz, "batch_size": BATCH[args.imgsz], "seed": args.seed,
        "device": 0, "visible_gpus_required": 1,
        "primary_sampler": {"positive_draws": N_POS, "negative_draws": N_NEG,
                            "hard_images": config.HARD_NEGATIVE_COUNT, "hard_weight": HARD_WEIGHT,
                            "other_negative_weight": 1.0, "negative_replacement": True},
        "augmentation_scope": "Weighted primary draws only; native mosaic auxiliary selection unchanged",
        "hf_repository": args.repo_id, "backup_every_epochs": BACKUP_EVERY,
        "recovery": {"run": args.resume_hf_run, "completed_epoch": args.resume_epoch, "revision": args.hf_revision},
        "expected_versions": {"ultralytics": config.EXPECTED_ULTRALYTICS, "torch": config.EXPECTED_TORCH},
    }


def source_files():
    # Snapshot the code actually executed, even with a separate data workspace.
    return {"train_hard_negatives.py": Path(__file__).resolve(), "config.py": Path(config.__file__).resolve()}


def main(argv=None):
    args, project = resolve_arguments(argv)
    resolved = resolved_configuration(args, project)
    if args.print_config:
        print(json.dumps(resolved, indent=2, allow_nan=False))
        return
    old_cwd = Path.cwd()
    try:
        run_training(args, project, resolved)
    finally:
        os.chdir(old_cwd)


def run_training(args, project, resolved):
    check(project.project_root.is_dir(), "The configured project workspace does not exist.")
    # For fresh runs, reject changed inputs before importing GPU/Hub libraries.
    prepared = None
    if not args.resume_hf_run:
        candidate_bytes = args.candidates.read_bytes()
        check(hashlib.sha256(candidate_bytes).hexdigest() == args.candidate_sha256, "Candidate CSV changed.")
        prepared = load_pool(candidate_bytes, args.dataset, args.manifest)
    project.activate_environment()
    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    import torch
    import ultralytics
    import yaml
    from ultralytics import YOLO
    from huggingface_hub import HfApi, CommitOperationAdd, hf_hub_download
    check(ultralytics.__version__ == config.EXPECTED_ULTRALYTICS, f"Use ultralytics=={config.EXPECTED_ULTRALYTICS}.")
    check(str(torch.__version__) == config.EXPECTED_TORCH, f"Use the existing pinned torch=={config.EXPECTED_TORCH} environment.")
    check(torch.cuda.is_available() and torch.cuda.device_count() == 1, "Expose exactly one GPU using CUDA_VISIBLE_DEVICES.")
    batch, completed, source = BATCH[args.imgsz], 0, None
    torch.cuda.set_device(0)
    api = HfApi()
    api.whoami()

    def upload(files, message):
        for attempt in range(config.UPLOAD_RETRIES):
            try:
                check(api.repo_info(args.repo_id).private is True, "Repository is no longer private.")
                result = api.create_commit(repo_id=args.repo_id, repo_type="model", commit_message=message,
                    operations=[CommitOperationAdd(path_in_repo=remote, path_or_fileobj=str(local)) for remote, local in files])
                print(f"\nHF BACKUP COMPLETE: {result.commit_url}", flush=True)
                return
            except Exception as exc:
                print(f"HF upload attempt {attempt + 1}/{config.UPLOAD_RETRIES} failed: {type(exc).__name__}", flush=True)
                if attempt < config.UPLOAD_RETRIES - 1:
                    time.sleep((15, 60)[attempt])
        raise RuntimeError("HF upload failed. Training stopped; local checkpoints remain.")

    # Optional recovery of an interrupted HN run, not initialization from a baseline.
    if args.resume_hf_run:
        check(api.repo_info(args.repo_id).private is True, "Recovery repository must be private.")
        def download(filename):
            return Path(hf_hub_download(args.repo_id, f"runs/{args.resume_hf_run}/{filename}", revision=args.hf_revision))
        old_env = json.loads(download("environment.json").read_text())
        check((old_env["imgsz"], old_env["batch_size"], old_env["seed"], old_env["recipe"], old_env["hard_weight"]) ==
              (args.imgsz, batch, args.seed, RECIPE, HARD_WEIGHT), "Recovery configuration differs.")
        if "source_sha256" in old_env:
            check(isinstance(old_env["source_sha256"], dict) and set(old_env["source_sha256"]) == set(source_files()),
                  "Unexpected recovery source fingerprint inventory.")
            for name, expected in old_env["source_sha256"].items():
                check(sha(download(name)) == expected, f"Recovery source fingerprint differs: {name}")
        candidate_bytes = download("candidates.csv").read_bytes()
        source = {f: download(f"epoch_{args.resume_epoch:03d}/{f}") for f in (*SNAPSHOT_FILES, "checkpoint_info.json")}
        source_info = json.loads(source["checkpoint_info.json"].read_text())
        check(source_info["completed_epoch"] == args.resume_epoch and source_info["run_name"] == args.resume_hf_run,
              "Wrong recovery snapshot.")
        for f in SNAPSHOT_FILES:
            check(sha(source[f]) == source_info["files_sha256"][f], f"Corrupted recovery file: {f}")
        completed = args.resume_epoch
    cohort, pool, pool_bytes = prepared if prepared is not None else load_pool(candidate_bytes, args.dataset, args.manifest)
    pool_sha = hashlib.sha256(pool_bytes).hexdigest()
    if source:
        check(source_info["manifest_sha256"] == MANIFEST_SHA and source_info["hard_pool_sha256"] == pool_sha and
              old_env["candidate_csv_sha256"] == hashlib.sha256(candidate_bytes).hexdigest(), "Recovery pool differs.")

    data_yaml = yaml.safe_load((args.dataset / "data.yaml").read_text())
    check(data_yaml["train"] == "images/train" and data_yaml["val"] == "images/val", "Unexpected data split paths.")
    check(len(data_yaml["names"]) == 1 and str(data_yaml["names"][0]).lower() == "fracture", "Expected fracture-only data.")
    data_yaml["path"] = str(args.dataset)
    api.create_repo(args.repo_id, repo_type="model", private=True, exist_ok=True)
    check(api.repo_info(args.repo_id).private is True, "HF backup repository must be private.")

    # Flat output layout; data.yaml is rebased ONLY in this run, for the new machine.
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%fZ")
    run = args.output_root / f"hardneg_yolo26s_{args.imgsz}_{EPOCHS}epochs_seed{args.seed}_direct100_{stamp}"
    run.mkdir(parents=True, exist_ok=False)
    check(not api.file_exists(args.repo_id, f"runs/{run.name}/environment.json"), "Remote run already exists.")
    (run / "weights").mkdir()
    (run / "recovery").mkdir()
    (run / "candidates.csv").write_bytes(candidate_bytes)
    (run / "hard_negative_pool.csv").write_bytes(pool_bytes)
    for filename in ("split_manifest.csv", "dataset_info.json"):
        shutil.copy2(args.dataset / filename, run / filename)
    source_hashes = {}
    for name, original in source_files().items():
        shutil.copy2(original, run / name)
        source_hashes[name] = sha(run / name)
        check(source_hashes[name] == sha(original), f"Source copy changed: {name}")
    resolved.update(run_name=run.name, run_dir=str(run), source_sha256=source_hashes,
                    candidate_csv_sha256=hashlib.sha256(candidate_bytes).hexdigest())
    dump(run / "resolved_config.json", resolved)
    (run / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False))
    audit = run / "sampler_audit.csv"
    env = {"run_name": run.name, "recipe": RECIPE, "status": "initialized", "imgsz": args.imgsz,
           "batch_size": batch, "seed": args.seed, "hard_weight": HARD_WEIGHT, "hard_candidates": len(pool),
           "manifest_sha256": MANIFEST_SHA, "hard_pool_sha256": pool_sha,
           "candidate_csv_sha256": hashlib.sha256(candidate_bytes).hexdigest(),
           "pool_policy": "Annotation-defined candidates; blank review is not clinical confirmation; exclude/uncertain omitted.",
           "augmentation_scope": "Weighted primary draws; native mosaic auxiliary image selection unchanged.",
           "reviewed_keep_count": sum(r["review"] == "keep" for r in pool),
           "python": sys.version, "python_executable": sys.executable,
           "torch": str(torch.__version__), "cuda": torch.version.cuda,
           "ultralytics": ultralytics.__version__, "gpu": torch.cuda.get_device_name(0),
           "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
           "backup_every_epochs": BACKUP_EVERY, "resume_source": args.resume_hf_run, "resume_commit": args.hf_revision,
           "source_sha256": source_hashes,
           "resume_source_sha256": old_env.get("source_sha256") if source else None,
           "test_evaluated": False}
    dump(run / "environment.json", env)
    common = ("environment.json", "candidates.csv", "hard_negative_pool.csv", "split_manifest.csv",
              "dataset_info.json", "data.yaml", "train_hard_negatives.py", "config.py", "resolved_config.json")
    def common_files():
        check(sha(run / "hard_negative_pool.csv") == pool_sha and
              sha(run / "candidates.csv") == env["candidate_csv_sha256"] and
              sha(run / "split_manifest.csv") == MANIFEST_SHA, "Frozen run inputs changed.")
        check(all(sha(run / name) == expected for name, expected in source_hashes.items()), "Run source snapshots changed.")
        return [(f"runs/{run.name}/{f}", run / f) for f in common]
    upload(common_files(), f"{run.name}: initialize frozen HN experiment")
    hard = {r["image_id"] for r in pool}
    Trainer, begin, finish = make_trainer(cohort, hard, args.seed, batch, audit)
    print(f"Run: {run}\nPool: {len(hard)} images; SHA256 {pool_sha}\nMatched batch: {batch}", flush=True)
    os.chdir(project.project_root)

    while completed < EPOCHS:
        target, start = EPOCHS, completed
        if source:
            check_history(source["results.csv"], completed)
            check_history(source["sampler_audit.csv"], completed)
            state = torch.load(source["last.pt"], map_location="cpu", weights_only=False)
            check(state["epoch"] == completed - 1 and all(state.get(k) is not None for k in
                  ("optimizer", "scaler", "ema", "updates", "best_fitness")), "Full-state recovery checkpoint required.")
            saved = state["train_args"]
            check(saved["epochs"] == EPOCHS and source_info["stage_target_epochs"] == EPOCHS,
                  "Recovery must use this single 100-epoch schedule, not a 50-epoch continuation.")
            check((saved["imgsz"], saved["batch"], saved["seed"]) == (args.imgsz, batch, args.seed), "Resume recipe changed.")
            saved.update(epochs=target, project=str(run.parent), name=run.name, save_dir=str(run),
                         exist_ok=True, data=str(run / "data.yaml"), device=0)
            resume_path = run / "recovery" / "resume.pt"
            torch.save(state, resume_path)
            del state
            for f, destination in (("best.pt", run / "weights/best.pt"), ("results.csv", run / "results.csv"),
                                   ("sampler_audit.csv", audit)):
                shutil.copy2(source[f], destination)
            model = YOLO(str(resume_path))
        else:
            model = YOLO("yolo26s.pt")  # Official COCO pretrained model, not a trained baseline.
            env["initial_checkpoint_sha256"] = sha(model.ckpt_path)
        env.update(status="training", stage_target_epochs=target)
        dump(run / "environment.json", env)
        backed_up = set()

        def verify(t):
            check(Path(t.save_dir).resolve() == run.resolve() and t.start_epoch == start and t.epochs == target,
                  "Wrong training folder or epoch range.")
            check(t.batch_size == batch and bool(t.amp), "Matched batch and AMP are required.")
            print(f"\nVERIFIED: epochs {start + 1}-{target}, {args.imgsz}px, batch {batch}", flush=True)

        def backup(t):
            epoch = int(t.epoch) + 1
            if epoch % BACKUP_EVERY or epoch in backed_up:
                return
            recovery = run / "recovery"
            # Copy FULL state before final_eval strips optimizer/EMA from live weights.
            for f, original in (("last.pt", t.last), ("best.pt", t.best), ("results.csv", t.csv),
                                ("args.yaml", run / "args.yaml"), ("sampler_audit.csv", audit)):
                shutil.copy2(original, recovery / f)
            check_history(recovery / "results.csv", epoch)
            check_history(recovery / "sampler_audit.csv", epoch)
            state = torch.load(recovery / "last.pt", map_location="cpu", weights_only=False)
            check(state["epoch"] == epoch - 1 and all(state.get(k) is not None for k in
                  ("optimizer", "scaler", "ema", "updates", "best_fitness")), "Backup lost resume state.")
            del state
            dump(recovery / "checkpoint_info.json", {"run_name": run.name, "completed_epoch": epoch,
                 "stage_target_epochs": target, "manifest_sha256": MANIFEST_SHA, "hard_pool_sha256": pool_sha,
                 "files_sha256": {f: sha(recovery / f) for f in SNAPSHOT_FILES}})
            upload(common_files() + [(f"runs/{run.name}/epoch_{epoch:03d}/{f}", recovery / f)
                                    for f in (*SNAPSHOT_FILES, "checkpoint_info.json")], f"{run.name}: epoch {epoch}")
            backed_up.add(epoch)

        model.add_callback("on_train_start", verify)
        model.add_callback("on_train_epoch_start", begin)
        model.add_callback("on_train_epoch_end", finish)
        model.add_callback("on_model_save", backup)
        if source:
            model.train(trainer=Trainer, resume=True, device=0, batch=batch, save_dir=str(run))
        else:
            model.train(trainer=Trainer, data=str(run / "data.yaml"), imgsz=args.imgsz, epochs=EPOCHS,
                        batch=batch, patience=0, optimizer="SGD", lr0=0.01, lrf=0.01, momentum=0.937,
                        weight_decay=0.0005, nbs=64, amp=True, device=0, seed=args.seed, deterministic=True,
                        workers=2, cache=False, val=True, split="val", nms=None, rect=False,
                        cos_lr=False, close_mosaic=10, project=str(run.parent), name=run.name,
                        exist_ok=True, save=True, save_period=-1, plots=True)
        check(target in backed_up, "Stage did not finish with a full-state HF backup.")
        source = {f: run / "recovery" / f for f in SNAPSHOT_FILES}
        completed = target
        del model
        gc.collect()
        torch.cuda.empty_cache()
    env.update(status="complete", completed_epochs=EPOCHS)
    dump(run / "environment.json", env)
    # Final inference weights, history and plots, in the same flat layout as baseline.
    final = [p for p in run.iterdir() if p.suffix in {"png", "jpg"}]
    final += [run / f for f in ("weights/best.pt", "weights/last.pt", "results.csv", "args.yaml", "sampler_audit.csv")]
    upload(common_files() + [(f"runs/{run.name}/{p.relative_to(run).as_posix()}", p) for p in final], f"{run.name}: complete")
    print(f"\nTRAINING COMPLETE: {run}\nBest: {run / 'weights/best.pt'}\nTest split was not evaluated.")


if __name__ == "__main__":
    main()
