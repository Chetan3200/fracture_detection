# Fracture detection repository

Reproducible research workflow for pediatric wrist-fracture localization with **YOLO26s** and **MedGemma**. It includes patient-isolated data preparation, a four-configuration YOLO ablation, QLoRA fine-tuning, checkpoint recovery, evaluation, and image inference. Research use only; model scores are not calibrated diagnostic probabilities.

This is the authoritative root guide. Use root `evaluate.py` and `infer.py`; the
backend files in `evaluation_scripts/` preserve the historical implementation
and source hashes, including their original invocation examples. Generated runs,
weights, caches, and downloaded data are ignored by Git.

**Review paths:** inspect [RESULTS.md](RESULTS.md), run the offline tests in
section 7, then follow sections 1–5 for data restore, training, and validation.
To replay existing models without training, use the pinned HF commands in section 5.

## Recorded experiment results

See **[RESULTS.md](RESULTS.md)** for the complete four-way YOLO ablation for seeds
**42 and 43**, MedGemma fine-tuning results, exact validation cutoffs, confidence
intervals, recorded hardware measurements, and source-run provenance.

| YOLO26s recorded result | Split | AP50:95 (%) | F1 (%) | FP/image |
|---|---|---:|---:|---:|
| HN 960, seed 42 | Validation | 56.01 | 90.46 | 0.0615 |
| HN 960, seed 43 | Validation | 56.01 | 90.18 | 0.0832 |
| HN 960, seed 42 | Test | 56.82 | 90.85 | 0.0630 |

| MedGemma, seed 42 | Precision (%) | Recall (%) | F1 (%) | FP/image |
|---|---:|---:|---:|---:|
| Validation, all generated boxes | 66.78 | 50.07 | 57.23 | 0.2352 |

YOLO uses validation-selected confidence cutoffs. MedGemma's updated table uses
all generated boxes and score-independent one-to-one matching at IoU >= 0.50;
AP and recalculated confidence intervals are not reported for this offline rescore.
Section 5 includes the exact command for this result.

HN960 has the highest recorded validation F1 in both seeds; seed-43 baseline
640 is almost tied at 90.17%. Seed-42 baselines used 50-to-100-epoch continuations;
all four seed-43 configurations used continuous 100-epoch schedules. The only
completed held-out test is seed-42 HN960; it must not be reused for tuning.
MedGemma has one seed-42 validation result, not a seed-43 or test result.
Original reports are preserved locally under ignored `runs/` directories.

## Private Hugging Face repositories

| Purpose | Repository ID | Configuration setting |
|---|---|---|
| Trained YOLO26 checkpoints, seeds 42 and 43 | [`Crimson-Dawn/grazpedwri-yolo26-checkpoints`](https://huggingface.co/Crimson-Dawn/grazpedwri-yolo26-checkpoints) | `HF_REPO_ID` |
| Fine-tuned MedGemma checkpoints | [`Crimson-Dawn/medgemma-fracture-checkpoints`](https://huggingface.co/Crimson-Dawn/medgemma-fracture-checkpoints) | `MEDGEMMA_HF_REPO_ID` |
| Frozen prepared dataset and splits | [`Crimson-Dawn/grazpedwri-frozen-splits`](https://huggingface.co/datasets/Crimson-Dawn/grazpedwri-frozen-splits) | `DATASET_HF_REPO_ID` |

All three repositories are private and require an authorized Hugging Face login.
The first two are **model** repositories; the third is a **dataset** repository.
[RESULTS.md](RESULTS.md#6-source-records-and-reproducibility) records the immutable
checkpoint revisions used for each reported result.

The dataset's `LATEST_DATA_BACKUP.json` at revision
`e3aec7e0d16358c283ef850730dd42bdd96b151b` records a completed snapshot:
`snapshots/c82f835e73885b53df8284c0ae28cf750a815c2557de554a315bfbc34800d618`.
Its manifest SHA256 matches the frozen data contract below. These identify the
existing backup. The restore command below verifies its inventory and file checksums.

## 1. Before running anything

### Access and runtime prerequisites

- Access to this private GitHub repository and the three private HF repositories above.
- Accept the access terms for [`google/medgemma-1.5-4b-it`](https://huggingface.co/google/medgemma-1.5-4b-it) with the same HF account before using MedGemma.
- Linux, a CUDA-capable NVIDIA GPU, and an **already-provisioned compatible Python environment** for training/inference. MedGemma additionally requires BF16 support. The recorded run used Python **3.12.14**, CUDA **12.8**, and RTX 5090 hardware; other hardware must pass the smoke check.
- Python **3.9+** alone is enough for configuration previews, control-flow tests, and offline MedGemma rescoring. It is not sufficient by itself for GPU execution.

```bash
git clone git@github.com:Chetan3200/fracture_detection.git
cd fracture_detection
# Keep long GPU jobs in this terminal session; detach with Ctrl-b, then d.
tmux new-session -s fracture -c "$PWD"
```

Run the following setup **inside that tmux shell**, from the cloned repository root:

```bash
export ROOT="$PWD"
# Use an existing environment. Override PY if it lives outside this checkout.
# Established training server: /workspace/ce-assignment/.venv/bin/python
export PY="${PY:-$ROOT/.venv/bin/python}"
export PREDICTION_PYTHON="$PY" MEDGEMMA_PYTHON="$PY"
export PATH="$(dirname "$PY"):$PATH"

python3 -B train_baseline.py --imgsz 640 --print-config
"$PY" -m pip check
hf auth login       # Interactive login; do not put a token in a command or file.
hf auth whoami
```

Use private repositories you can **write** to for new training backups. The
recorded repositories above are also read-only sources for checkpoint replay:

```bash
export HF_REPO_ID=YOUR_HF_ACCOUNT/private-yolo-runs
export MEDGEMMA_HF_REPO_ID=YOUR_HF_ACCOUNT/private-medgemma-runs
```

`requirements.txt` specifies direct dependencies, not a full environment lock.
Exact historical protocol replay requires the recorded versions below; an OpenCV
module version is not a wheel/build identifier. Preserve the existing matching
environment rather than replacing Torch/CUDA or guessing another OpenCV build.
The scripts never install packages automatically and do not need FlashAttention.

```bash
# Runtime check only: no model loading, downloads, or training.
"$PY" - <<'PY'
import torch, numpy, pandas, cv2, ultralytics
import transformers, peft, accelerate, bitsandbytes, huggingface_hub
expected = {
    'torch': (torch.__version__, '2.11.0+cu128'),
    'numpy': (numpy.__version__, '2.5.2'),
    'pandas': (pandas.__version__, '3.0.6'),
    'opencv': (cv2.__version__, '5.0.0'),
    'ultralytics': (ultralytics.__version__, '8.4.152'),
    'transformers': (transformers.__version__, '4.57.6'),
    'peft': (peft.__version__, '0.18.1'),
    'accelerate': (accelerate.__version__, '1.12.0'),
    'bitsandbytes': (bitsandbytes.__version__, '0.49.2'),
    'huggingface_hub': (huggingface_hub.__version__, '0.36.2'),
}
for name, (actual, wanted) in expected.items():
    print(f'{name}: {actual} (expected {wanted})')
    if actual != wanted:
        raise SystemExit(f'Use the recorded environment: {name} differs')
if not torch.cuda.is_available():
    raise SystemExit('CUDA is unavailable')
print('CUDA:', torch.version.cuda, 'GPUs:', torch.cuda.device_count())
print('GPU 0:', torch.cuda.get_device_name(0), 'BF16:', torch.cuda.is_bf16_supported())
PY
```

If a prerequisite is missing, provision the matching runtime or obtain access
before the GPU steps. A requirements-only installation is not an exact replay of
the saved software identity. A protocol also checks checkpoint and source hashes.

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

**Preferred path: restore the recorded snapshot.** The dataset repository must be private. Backup writes an inventory, checks the
manifest and file checksums, does not overwrite a completed snapshot, and prints
the immutable HF revision and snapshot ID. Restore verifies the completion
record, inventory, shards, checksums, and manifest and never overwrites an
existing dataset.

```bash
# Preview either command first.
python3 -B backup_data_to_hf.py --print-config
python3 -B download_data_from_hf.py --print-config

# Restore the recorded frozen snapshot into DATA_DIR (default: data/).
"$PY" download_data_from_hf.py \
  --repo-id Crimson-Dawn/grazpedwri-frozen-splits \
  --snapshot c82f835e73885b53df8284c0ae28cf750a815c2557de554a315bfbc34800d618 \
  --revision e3aec7e0d16358c283ef850730dd42bdd96b151b

# Optional: back up your completed prepared dataset to a private destination.
"$PY" backup_data_to_hf.py --repo-id YOUR_HF_ACCOUNT/YOUR_PRIVATE_DATASET
```

Without `--revision`, restore resolves HEAD once and records the resolved commit
in its receipt. Preserve every backup/restore receipt with the run it supports.

### Build locally only when a prepared snapshot is unavailable

Use this **instead of restore**, not after it: prepare YOLO first, then MedGemma
from that YOLO dataset. The original source is Kaggle
`jasonroggy/grazpedwri-dx/versions/1`; use an authorized KaggleHub download/cache.
The frozen `split_manifest.csv` is already tracked in this repository. Both
preparers validate it and refuse to replace existing prepared directories.
Images without fracture boxes but with AO classification, visible-fracture, or
uncertain-diagnosis signals were excluded; negatives are annotation-defined.

```bash
$PY data_prep/prepare_data.py
$PY data_prep/prepare_medgemma.py
```

YOLO preparation may use KaggleHub's existing cache. Keep that cache: locally
prepared YOLO images are symlinks to it. The portable HF snapshot materializes
image files, so a restored snapshot does not require the original cache path.

After either route, the default layout is:

```text
data/
├── grazpedwri_yolo/       # data.yaml, images/{train,val,test}, labels/{train,val,test}
└── grazpedwri_medgemma/   # RGB images, prompt.txt, train/val JSONL, answer-free generation inputs
```

Check prepared metadata before training (offline, no image decoding):

```bash
python3 -B - <<'PY'
import json
from config import load_config, sha256, MANIFEST_SHA256, SPLIT_COUNTS
c = load_config()
if sha256(c.manifest_path) != MANIFEST_SHA256:
    raise SystemExit('Root manifest mismatch')
for folder in (c.yolo_data_dir, c.medgemma_data_dir):
    if sha256(folder / 'split_manifest.csv') != MANIFEST_SHA256:
        raise SystemExit(f'Manifest mismatch: {folder}')
    info = json.loads((folder / 'dataset_info.json').read_text())
    if info['manifest_sha256'] != MANIFEST_SHA256 or info['split_counts'] != SPLIT_COUNTS:
        raise SystemExit(f'Dataset metadata mismatch: {folder}')
    print(folder, 'verified:', info['included_images'], 'included images')
PY
```

The training loaders additionally check input files and label consistency. The
backup contains all splits for archival completeness; training loaders use only
the training split, with validation for selection.

## 3. YOLO training

Use one continuous 100-epoch schedule, seed `43`, on one visible GPU.
Seeds and deterministic settings do not guarantee bitwise-identical GPU results. The fixed defaults are 640px/batch 35 or 960px/batch 14. Do not allow an
OOM fallback to silently change batch size. Each run gets a new timestamped
folder and records configuration, source snapshots, manifest, environment, and
private backup receipts. The shared optimizer recipe is SGD, LR **0.01** with
linear decay to a **0.01 multiplier**, momentum **0.937**, weight decay **0.0005**,
nominal batch **64**, AMP enabled, and mosaic disabled for the final **10** epochs.
Early stopping is disabled; `best.pt` is selected by trainer validation fitness.

```bash
# Run inside tmux. Baselines: ordinary sampling, private backups every 10 epochs.
CUDA_VISIBLE_DEVICES=0 "$PY" -u train_baseline.py --imgsz 640 --seed 43 --epochs 100
CUDA_VISIBLE_DEVICES=0 "$PY" -u train_baseline.py --imgsz 960 --seed 43 --epochs 100

# For a local-only baseline, explicitly add --no-hf-backup to its command.
```

Set `HF_REPO_ID=YOUR_OWNER/YOUR_PRIVATE_MODEL_REPO` in the environment or `.env`,
or pass `--repo-id`. Baseline permits `--no-hf-backup`; hard-negative training
does not.

### Hard-negative experiment

`train_hard_negatives.py` is a separate fixed experiment, not a mining loop. It
uses the fixed 178-image pool, primary-draw weight `3`, and ranked CSV
SHA256:

```text
5af4772bf275a4e84a815670d2988fb0408119ea05e0b2ffe0cb3251fdf1e66d
```

The pool was mined **once from the seed-42 baseline640 checkpoint**, deliberately
reused for both training seeds: `mine_hard_negatives.py` pins repository revision
`d104d0498236676d0882942a37ee1c353d80cfda`, runs only the 3,961 training negatives,
and selects maximum fracture confidence >= 0.10, capped at 20% of negatives and
two images per patient. The frozen CSV is tracked; its historical `image_path`
column is provenance, while training resolves images by ID in the current dataset.
Do not re-mine or edit it to reproduce this ablation.

Each epoch includes 9,479 positive primary draws exactly once and 3,961 negative
draws with replacement (hard weight 3, other weight 1), then shuffles the 13,440
draws. Mosaic's auxiliary image selection is unchanged. HN training starts fresh
from COCO weights for 100 continuous epochs, at 640/35 or 960/14, and creates
private full-state HF backups every 10 epochs.

For the historical seed-42 baseline numbers, evaluate the recorded checkpoints
in RESULTS.md. Changing the current trainer to `--seed 42` creates a **new direct
100-epoch baseline**, not the historical 50-to-100 continuation. HN seed-42 runs
use the same commands below with `--seed 42`.

```bash
# These complete the four-way seed-43 ablation. --imgsz is required.
CUDA_VISIBLE_DEVICES=0 "$PY" -u train_hard_negatives.py --imgsz 640 --seed 43
CUDA_VISIBLE_DEVICES=0 "$PY" -u train_hard_negatives.py --imgsz 960 --seed 43
```

Recovery is only for an interrupted HN run, into a new local folder, and must
pin the saved 10-epoch boundary and exact HF commit:

```bash
CUDA_VISIBLE_DEVICES=0 "$PY" train_hard_negatives.py --imgsz 960 --seed SAVED_SEED \
  --resume-hf-run RUN_NAME --resume-epoch COMPLETED_10_EPOCH_BOUNDARY \
  --hf-revision IMMUTABLE_40_CHAR_COMMIT
```

It validates the saved source recipe, pool, manifest, seed, image size, batch,
and schedule before resuming. It is not a way to change an experiment.
Only load trusted checkpoint archives: PyTorch `.pt`/optimizer state can contain
pickle objects; a matching checksum establishes identity, not trust in its author.

## 4. MedGemma training

MedGemma uses `google/medgemma-1.5-4b-it` at immutable revision
`91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b`. The frozen recipe is NF4
double-quantization, BF16 compute, language-only LoRA, microbatch 1 per GPU,
three epochs, seed 42, and recovery backups every 100 optimizer updates.
LoRA rank/alpha/dropout are **16/32/0.05** on language-model attention and MLP
projections; the vision encoder/projector remain frozen. Optimizer: fused AdamW,
LR **1e-4**, weight decay **0.01**, cosine decay, **3%** warmup, gradient clipping
**1.0**, and non-reentrant gradient checkpointing. Loss covers assistant answer
tokens only. The best adapter minimizes full-validation answer-token loss.

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

# Recorded recipe: two GPUs, accumulation 8, effective batch 16.
# Run sequentially in tmux. && prevents full training after a failed smoke/gate.
"$PY" -u medgemma_training/run_medgemma.py smoke --devices 0,1 --grad-accum 8 &&
"$PY" medgemma_training/run_medgemma.py train --devices 0,1 --grad-accum 8 --check-ready &&
"$PY" -u medgemma_training/run_medgemma.py train --devices 0,1 --grad-accum 8
```

Set `MEDGEMMA_HF_REPO_ID` to a private write destination, or use `--repo-id`.
The launcher uses the existing environment selected in section 1; `--python`
can override it. For one GPU, use `--devices 0 --grad-accum 16` for **all three**
commands. Different GPU counts require their own matching smoke. `--check-ready`
checks local smoke receipts; the actual launch checks the live backup destination.
Use this launcher for new runs; direct trainer invocation is reserved for exact
archived recovery, where the saved repository and all saved flags must be retained.

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
CUDA_VISIBLE_DEVICES=SAVED_DEVICE_IDS "$PY" -u -m torch.distributed.run \
  --standalone --nnodes=1 --nproc_per_node=SAVED_RANK_COUNT \
  /ORIGINAL/RUN/training_script.py --project-root "$ROOT" \
  --data-root "$ROOT/data/grazpedwri_medgemma" \
  --output /ORIGINAL/RUN --resume /ORIGINAL/RUN/checkpoint-SAVED_STEP \
  --hf-repo-id SAVED_PRIVATE_REPO \
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

### Replay the recorded private checkpoints

These are complete **validation** examples using immutable HF sources. Run in
tmux with the restored dataset and matching runtime. Outputs must be new paths;
for a repeat, choose a different output directory. Add `--smoke` with a separate
output first if checking a newly configured machine.

```bash
CUDA_VISIBLE_DEVICES=0 "$PY" -u evaluate.py yolo26 --hf \
  --hf-repo Crimson-Dawn/grazpedwri-yolo26-checkpoints \
  --hf-revision f909d6feb3ec3022ed1931447d14e28e819ec452 \
  --hf-filename runs/hardneg_yolo26s_960_100epochs_seed42_20260918_092553_812045Z/epoch_100/best.pt \
  --imgsz 960 --batch 8 --output runs/replay_yolo_hn960_seed42_val

CUDA_VISIBLE_DEVICES=0 "$PY" -u evaluate.py medgemma --hf \
  --hf-repo Crimson-Dawn/medgemma-fracture-checkpoints \
  --hf-revision bec122fe4557cf6af3468c6e5853ce0fa66d695c \
  --hf-run medgemma_train_2gpu_20260917_134859 \
  --output runs/replay_medgemma_val
```

The source records in RESULTS.md give the revisions and exact checkpoint paths
for all eight YOLO configurations. Substitute the matching filename, revision,
and image size to evaluate another configuration. For newly trained local runs,
use the `best.pt` path printed by that trainer and its matching image size.
Evaluation output contains `predictions.jsonl`, `metrics.csv`, `subgroups.csv`,
`evaluation.json`, and the root launcher's `launcher_config.json`.

### Unfiltered MedGemma precision, recall, F1, and FP/image

The historical shared evaluator above remains unchanged: its `metrics.csv` uses
a token-score-selected cutoff. To obtain the **updated all-box results** reported
in this repository, run the separate standard-library-only command:

```bash
# Use predictions from the replay above, or the original saved evaluation directory.
python3 -B rescore_medgemma.py \
  --predictions runs/replay_medgemma_val/predictions.jsonl \
  --output runs/replay_medgemma_val_unfiltered
```

The sibling `evaluation.json` must be a completed MedGemma validation report
whose prediction checksum matches. The command verifies that every raw generated
box was retained, uses no token-score threshold, matches boxes one-to-one with
maximum cardinality at IoU >= 0.50, and writes a **new** `metrics.csv` plus
`rescore.json` with counts, hashes, runtime, and method identity. It does not run
the model or change the original files. No AP or confidence intervals are computed;
`rescore.json` is **not** a test/inference `--protocol` file.

For the exact archived point estimates, use the original predictions identified
by SHA256 in RESULTS.md; those local reports are not bundled in Git. The pinned
checkpoint command above regenerates validation predictions when the archived
files are unavailable. Hardware/software changes can affect newly generated
predictions, so retain their own provenance and do not overwrite old reports.

### Frozen test protocol

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
otherwise. Keep new validation reruns separate from the archived reports.

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
overwritten. The validation protocol gates checkpoint/settings/runtime and the
shared prediction-helper hashes; the separate inference entrypoint/common source
hashes are recorded in the inference output for provenance. MedGemma protocol
inference is token-score-filtered, unlike the offline all-box rescore above.

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
FRACTURE_MEDGEMMA_RESULTS="$PWD/runs/evaluation_medgemma_val_20260918_120322_626065Z" \
  python3 -B -m unittest discover -s tests -p test_medgemma_rescore.py -v
```

The evaluator's licensing and metric-reference attribution are in
`evaluation_scripts/LICENSE` and `evaluation_scripts/tests/reference_ultralytics.py`.
If DATA_DIR, RUNS_DIR, or CACHE_DIR use custom in-repo names, add matching ignore
rules or place them outside the checkout. Keep credentials out of logs and
artifacts. Preview/unit tests verify configuration and logic; use the runtime
preflight and matching smoke run to verify GPU, model loading, and backup access.
