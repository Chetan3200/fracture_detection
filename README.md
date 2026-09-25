# Fracture detection repository

Research workflow for fracture detection with YOLO26s and fine-tuned MedGemma. This repository is **not** a clinical product: prediction scores, including MedGemma token-likelihood scores, are not diagnostic probabilities or clinical claims.

This is the authoritative root guide. The shared evaluators live in `evaluation_scripts/`; root `evaluate.py` and `infer.py` are the supported configured frontends. Generated runs, weights, caches, and downloaded data are ignored by Git but intentionally remain local.

## 1. Before running anything

Work from the repository root. The established runtime is the existing
`/workspace/ce-assignment/.venv/bin/python`; do not create another environment,
run package installation, replace Torch/CUDA, add FlashAttention, or enable an
automatic installer. Long GPU jobs belong in `tmux`.

```bash
cd /workspace/ce-assignment
PY=/workspace/ce-assignment/.venv/bin/python
# Any Python 3.9+ is sufficient for this offline configuration preview.
python3 -B train_baseline.py --imgsz 640 --print-config
```

Use the installed, runtime-appropriate pinned dependencies for actual training,
evaluation, and inference. `requirements.txt` is a dependency list, not a full
lockfile: its NumPy, pandas, and OpenCV entries are currently unpinned. Saved
YOLO evaluation protocols record Torch `2.11.0+cu128`, NumPy `2.5.2`, pandas
`3.0.6`, OpenCV `5.0.0`, and Ultralytics `8.4.152`; do not invent an OpenCV
distribution/build tag. A saved report is usable only when its recorded
checkpoint, source hashes, and runtime identity still match.

Copy `.env.example` to an ignored `.env` only if local paths or private Hugging
Face destinations differ. Never put a token in a command, source file, or `.env`.
The configuration rule is:

```text
explicit CLI option > exported environment > root .env > code default
```

All relative configuration paths are rooted at the project root, not the shell
working directory. `--print-config` previews are offline and do not import ML
libraries, use a GPU, download data/models, or create outputs.

Useful non-secret settings are `DATA_DIR`, `RUNS_DIR`, `CACHE_DIR`,
`MANIFEST_PATH`, `CANDIDATES_PATH`, `HF_REPO_ID`, `MEDGEMMA_HF_REPO_ID`,
`DATASET_HF_REPO_ID`, `HF_HOME`, `KAGGLEHUB_CACHE`, `PREDICTION_DEVICE`, and
`PREDICTION_PYTHON`. Evaluation/inference use one logical CUDA device per
invocation; MedGemma training uses the separately configured rank count.
Leave `HF_HOME` unset
to retain the existing authorized Hugging Face cache/login unless a specific
existing cache is required.

## 2. Frozen data contract

Every supported workflow uses this immutable patient-isolated manifest:

```text
SHA256  1719f37f442512c3c4fcab8350bedc0f27ff2dcebc082761299912b897747034
split   positives  negatives
train      9479       3961
val        2045        850
test       2026        846
```

The intended split is 70/15/15 by patient. Excluded images exist in the manifest
but are not positive/negative training or evaluation examples. Never resplit,
move patients, regenerate the manifest, or use test data for training, threshold
selection, or tuning.

### Restore or back up the prepared snapshot

Use these root scripts, not the deprecated `data_prep/prepared_dataset_transfer.py`.
The dataset repository must be private. Backup writes an inventory, checks the
manifest and file checksums, does not overwrite a completed snapshot, and prints
the immutable HF revision and snapshot ID. Restore verifies the completion
record, inventory, shards, checksums, and manifest and never overwrites an
existing dataset.

```bash
# Preview either command first.
python3 -B backup_data_to_hf.py --print-config
python3 -B download_data_from_hf.py --print-config

# Back up a completed local prepared dataset.
$PY backup_data_to_hf.py --repo-id YOUR_OWNER/YOUR_PRIVATE_DATASET

# Restore exactly the receipt printed by backup.
$PY download_data_from_hf.py --repo-id YOUR_OWNER/YOUR_PRIVATE_DATASET \
  --snapshot SNAPSHOT_ID --revision IMMUTABLE_40_CHAR_COMMIT
```

Without `--revision`, restore resolves HEAD once and records the resolved commit
in its receipt. Preserve every backup/restore receipt with the run it supports.

### Build locally only when a prepared snapshot is unavailable

Prepare YOLO first, then prepare MedGemma separately from that YOLO dataset.
Both scripts verify the same frozen manifest and do not create a split.

```bash
$PY data_prep/prepare_data.py
$PY data_prep/prepare_medgemma.py
```

YOLO preparation may use KaggleHub's existing cache. Preserve that cache and its
symlinks: prepared YOLO images link to it. Set `KAGGLEHUB_CACHE` or put the
persistent cache outside the prepared dataset; do not copy/flatten links merely
to relocate the data.

## 3. YOLO training

Use one continuous 100-epoch schedule, seed `43`, on one visible GPU.
Seeds and deterministic settings do not guarantee bitwise-identical GPU results. The fixed defaults are 640px/batch 35 or 960px/batch 14. Do not allow an
OOM fallback to silently change batch size. Each run gets a new timestamped
folder and records configuration, source snapshots, manifest, environment, and
private backup receipts.

```bash
# Baseline: ordinary sampling. Private backups every 10 epochs by default.
tmux new-session -d -s yolo-baseline \
  'cd /workspace/ce-assignment && CUDA_VISIBLE_DEVICES=0 /workspace/ce-assignment/.venv/bin/python -u train_baseline.py --imgsz 640 --seed 43 --epochs 100; exec bash'

# A local-only baseline is explicit, not the default:
$PY train_baseline.py --imgsz 960 --seed 43 --epochs 100 --no-hf-backup
```

Set `HF_REPO_ID=YOUR_OWNER/YOUR_PRIVATE_MODEL_REPO` in the environment or `.env`,
or pass `--repo-id`. Baseline permits `--no-hf-backup`; hard-negative training
does not.

### Hard-negative experiment

`train_hard_negatives.py` is a separate fixed experiment, not a mining loop. It
uses the reviewed 178-image frozen pool, primary-draw weight `3`, and ranked CSV
SHA256:

```text
5af4772bf275a4e84a815670d2988fb0408119ea05e0b2ffe0cb3251fdf1e66d
```

Do not edit the ranked CSV, re-mine candidates, or use exploratory mining or
overlays as the active workflow. It is a 100-epoch continuous run at the matched
640/35 or 960/14 batch. HN requires a private HF destination and makes full-state
backup snapshots every 10 epochs.

```bash
# --imgsz is required; this is the fixed 960px HN experiment.
tmux new-session -d -s yolo-hn \
  'cd /workspace/ce-assignment && CUDA_VISIBLE_DEVICES=0 /workspace/ce-assignment/.venv/bin/python -u train_hard_negatives.py --imgsz 960 --seed 43; exec bash'
```

Recovery is only for an interrupted HN run, into a new local folder, and must
pin the saved 10-epoch boundary and exact HF commit:

```bash
$PY train_hard_negatives.py --imgsz 960 \
  --resume-hf-run RUN_NAME --resume-epoch COMPLETED_10_EPOCH_BOUNDARY \
  --hf-revision IMMUTABLE_40_CHAR_COMMIT
```

It validates the saved source recipe, pool, manifest, seed, image size, batch,
and schedule before resuming. It is not a way to change an experiment.

## 4. MedGemma training

MedGemma uses `google/medgemma-1.5-4b-it` at immutable revision
`91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b`. The frozen recipe is NF4
double-quantization, BF16 compute, language-only LoRA, microbatch 1 per GPU,
three epochs, seed 42, and recovery backups every 100 optimizer updates.

The launcher supports any positive GPU count on one machine using `--devices` or
`--num-gpus`. World size `N` and accumulation determine effective batch `N *
grad_accum`. The default accumulation is exactly `16/N` only for exact divisors
such as N=1,2,4,8,16. Otherwise declare it: `N=3 --grad-accum 4` means effective
batch 12, a different experiment from effective batch 16.

Run a same-signature smoke before full training. A smoke is exactly four updates
and four generation-format checks; full training performs 16 generation-format
checks. Full launch requires matching N, devices, accumulation, GPU/NCCL
settings, trainer/helper/shared-config fingerprints, prepared data, and recipe, plus
successful private checkpoint and final backup receipts from smoke.

```bash
# Offline plans; Python 3.9+ is enough for these only.
python3 -B medgemma_training/run_medgemma.py smoke --num-gpus 1 --print-config
python3 -B medgemma_training/run_medgemma.py smoke --num-gpus 3 --grad-accum 4 --print-config

# Launch matching smoke then full training in tmux.
tmux new-session -d -s medgemma-smoke \
  'cd /workspace/ce-assignment && ./.venv/bin/python -u medgemma_training/run_medgemma.py smoke --num-gpus 1; exec bash'
tmux new-session -d -s medgemma-train \
  'cd /workspace/ce-assignment && ./.venv/bin/python -u medgemma_training/run_medgemma.py train --num-gpus 1; exec bash'
```

Set `MEDGEMMA_HF_REPO_ID=YOUR_OWNER/YOUR_PRIVATE_MEDGEMMA_REPO`, or use
`--repo-id`. The launcher defaults to the existing root `.venv`; `--python` may
select another already-existing executable. It never creates an environment.

For recovery, use the archive's immutable commit, checksum, `latest.json`,
original absolute output path, exact saved sources, dataset/prompt/recipe,
world size, devices, accumulation, and GPU/NCCL settings. Restore into the
original absolute run directory and resume with the archived training script,
not current source substituted in its place. Read `latest.json` and its ZIP at
the SAME immutable repository commit; verify `archive_sha256` before extracting
into an empty restore destination. The remote index does not contain its own
commit SHA; a local successful upload receipt does. Preserve both best and latest
checkpoints. Restore the frozen data separately and load pickle-compatible
optimizer/RNG files only from your trusted archive.

For a current full-run archive, substitute the original paths and SAVED settings:

```bash
CUDA_VISIBLE_DEVICES=SAVED_DEVICE_IDS $PY -u -m torch.distributed.run \
  --standalone --nnodes=1 --nproc_per_node=SAVED_RANK_COUNT \
  /ORIGINAL/RUN/training_script.py --project-root /workspace/ce-assignment \
  --data-root /workspace/ce-assignment/data/grazpedwri_medgemma \
  --output /ORIGINAL/RUN --resume /ORIGINAL/RUN/checkpoint-SAVED_STEP \
  --epochs 3 --grad-accum SAVED_ACCUMULATION --backup-steps 100 --generation-samples 16
```

Keep original NCCL settings too. Older snapshots without `--project-root` need
their original CLI with explicit `--data-root`. Never change world size mid-run.
A completed adapter is for inference, not restarting the same three epochs.
`medgemma_training/archive/train_medgemma_original.py` is provenance only.

## 5. Evaluate a selected model

Both models evaluate against `DATA_DIR/grazpedwri_yolo`, not a separate MedGemma
evaluation split. Root wrappers add configuration receipts while leaving the
hash-pinned shared canonical helpers in `evaluation_scripts/` unchanged. They
verify checkpoint/source/software/data identity; do not bypass these checks.

```bash
# Offline frontend plans
python3 -B evaluate.py yolo26 --checkpoint runs/YOUR_RUN/weights/best.pt --print-config
python3 -B evaluate.py medgemma --checkpoint runs/YOUR_MEDGEMMA_RUN --print-config

# First inspect a four-image validation smoke, then perform complete validation.
CUDA_VISIBLE_DEVICES=0 $PY -u evaluate.py yolo26 \
  --checkpoint runs/YOUR_RUN/weights/best.pt --imgsz 640 --batch 8 --smoke
CUDA_VISIBLE_DEVICES=0 $PY -u evaluate.py yolo26 \
  --checkpoint runs/YOUR_RUN/weights/best.pt --imgsz 640 --batch 8

CUDA_VISIBLE_DEVICES=0 $PY -u evaluate.py medgemma \
  --checkpoint runs/YOUR_MEDGEMMA_RUN --smoke
CUDA_VISIBLE_DEVICES=0 $PY -u evaluate.py medgemma \
  --checkpoint runs/YOUR_MEDGEMMA_RUN
```

Use `--hf` with `--hf-repo`, immutable `--hf-revision`, and model-specific
`--hf-filename` (YOLO) or `--hf-run` (MedGemma) for a private HF checkpoint.
A full validation `evaluation.json` stores the selected threshold, checkpoint
provenance, manifest, prediction settings, helper/source hashes, software
identity and metrics. The separate `launcher_config.json` records frontend
configuration and code hashes for successful new runs; existing reports are never
rewritten. The strict filter is `score > cutoff`.

AP50 and AP50:95 use the collected score-ranked detections, not just the chosen
operating cutoff. Lesion recall, FP/image, precision, and F1 use that cutoff at
IoU 0.50. Validation maximizes lesion F1; exact ties choose the highest tested
cutoff. Confidence intervals resample whole patients and exclude training and
threshold-selection uncertainty. Benchmarks report measured hardware timings,
not estimated deployment performance.

The 10 existing saved reports, including the baseline reports, were audited for
source-fingerprint compatibility. That does not waive their recorded runtime,
checkpoint, data, and source-hash requirements; do not treat compatibility as a
claim that a new live GPU execution was performed.

Final test is permitted only after selecting the model and approving that
model's **complete matching validation** protocol. It inherits rather than
retunes image size, batch, and MedGemma max-new-tokens:

```bash
CUDA_VISIBLE_DEVICES=0 $PY -u evaluate.py yolo26 \
  --checkpoint runs/YOUR_RUN/weights/best.pt --split test \
  --protocol runs/YOUR_COMPLETE_VALIDATION/evaluation.json
```

The HN 960 seed-42 test has already been executed. It is no longer untouched:
do not rerun it, tune against it, or claim a new test result because of cleanup.
Other existing results remain validation-only unless their saved report says
otherwise. No accuracy values are asserted in this guide.

## 6. Predict images or folders without ground truth

Inference supports one image or a directory (optionally `--recursive`) in PNG,
JPG/JPEG, BMP, WEBP, or single-frame TIFF. It writes provenance, structured
predictions, and optional overlays. It does not compute metrics, select a
threshold, or require labels. DICOM, multi-frame TIFF, clinical windowing, and
clinical interpretation are unsupported.

```bash
# Prefer the selected model's matching complete validation protocol.
CUDA_VISIBLE_DEVICES=0 $PY -u infer.py yolo26 \
  --checkpoint runs/YOUR_RUN/weights/best.pt \
  --protocol runs/YOUR_COMPLETE_VALIDATION/evaluation.json \
  --source /path/to/image-or-folder

CUDA_VISIBLE_DEVICES=0 $PY -u infer.py medgemma \
  --checkpoint runs/YOUR_MEDGEMMA_RUN \
  --protocol runs/YOUR_COMPLETE_VALIDATION/evaluation.json \
  --source /path/to/image-or-folder

# Explicit exploratory-only cutoff, with no protocol:
$PY -u infer.py yolo26 --checkpoint runs/YOUR_RUN/weights/best.pt \
  --source /path/to/image.png --conf 0.25 --imgsz 640 --batch 8
```

A protocol must be a matching complete validation `evaluation.json`; it supplies
cutoff and prediction settings and rejects conflicts. Without a protocol,
`--conf` is mandatory and recorded as exploratory. It is not a calibrated
confidence. Outputs go to a new directory; existing directories are never
overwritten.

## 7. Tests and scope checks

These commands make no network calls. Root/MedGemma/inference control-flow tests
use the standard library; the metric suite additionally needs existing NumPy,
pandas, and Torch CPU dependencies. It forbids GPU benchmarking and does not load
models or install anything.

```bash
python3 -B -m unittest discover -s tests -v
python3 -B -m unittest discover -s medgemma_training/tests -v
python3 -B -m unittest discover -s inference_scripts/tests -v
# Checkpoint/model-helper tests also run without scientific packages:
python3 -B -m unittest discover -s evaluation_scripts/tests -p test_checkpoints.py -v
python3 -B -m unittest discover -s evaluation_scripts/tests -p test_medgemma_model.py -v
# Full evaluator metric suite in the existing equipped runtime:
$PY -B -m unittest discover -s evaluation_scripts/tests -v
# Optional read-only regression against saved reports/predictions:
FRACTURE_RESULTS="$PWD/runs" python3 -B -m unittest discover -s inference_scripts/tests -v
```

Do not add stale directory aliases, alternative virtual environments, new
licenses, or third-party source copies to make a command appear portable. The
existing `evaluation_scripts/LICENSE` and metric-reference provenance are retained.
The deprecated transfer implementation stays local but is excluded from the
Git candidate set. If DATA_DIR, RUNS_DIR, or CACHE_DIR use custom in-repo names,
add matching ignore rules or place them outside the checkout. Keep
credentials out of logs and generated artifacts. A successful preview or unit
test is not proof that GPU availability, model loading, private HF access, or a
live training/evaluation run has been tested.
