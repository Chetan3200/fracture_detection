# Ablation study and MedGemma fine-tuning results

Results for the completed experiments, not the native trainer's epoch metrics. YOLO point estimates and confidence intervals come from saved shared-evaluator `metrics.csv` reports; identities, operating points, and hardware measurements come from `evaluation.json`. **MedGemma's four localization metrics below were recalculated offline from all saved generated boxes, without token-score filtering.** No training, model inference, threshold search, or test evaluation was rerun. Original saved reports remain unchanged.

## At a glance

- **YOLO HN 960 has the highest validation F1 in both seeds:** 90.46% for seed 42 and 90.18% for seed 43. Seed-43 baseline 640 is almost tied at 90.17%; this is not evidence of a decisive improvement.
- **The completed held-out test result is seed-42 YOLO HN 960:** AP50:95 56.82%, F1 90.85%, and 0.0630 FP/image, using its frozen validation protocol.
- **Fine-tuned MedGemma, seed 42, unfiltered:** validation precision 66.78%, recall 50.07%, F1 57.23%, and 0.2352 FP/image at IoU >= 0.50. These use all generated boxes and score-independent matching, not a token-score-selected operating point. MedGemma AP values are omitted.
- No seed-43 YOLO test report, MedGemma seed-43 result, or MedGemma test report was found in the available records. These must not be filled in from another seed or split.

## Evaluation scope and units

| Split | Images | Patients | Positive images | Negative images | Fracture boxes |
|---|---:|---:|---:|---:|---:|
| Validation | 2,895 | 893 | 2,045 | 850 | 2,734 |
| Test | 2,872 | 893 | 2,026 | 846 | 2,671 |

- All experiments use the frozen patient-isolated GRAZPEDWRI-DX split. Manifest SHA256: `1719f37f442512c3c4fcab8350bedc0f27ff2dcebc082761299912b897747034`.
- AP, recall, precision, and F1 are shown as **percentages**; FP/image is a **count rate over all evaluated images**, not a percentage or negative-image-only error rate. Higher is better except FP/image.
- AP50 uses IoU 0.50; AP50:95 averages over IoU 0.50:0.05:0.95. AP uses score-ranked collected detections, not just boxes passing the selected operating cutoff.
- **YOLO:** recall, precision, F1, and FP/image use the selected operating cutoff and lesion matching at IoU >= 0.50. Validation selects maximum lesion F1, choosing the highest tested cutoff for exact ties. Filtering is strictly `score > cutoff`. **Updated MedGemma:** no score filtering; maximum-cardinality one-to-one matching at IoU >= 0.50, independent of token scores.
- Original evaluator reports use 1,000 whole-patient bootstrap replicates, bootstrap seed 2026, and 95% percentile CIs. These intervals hold the model and cutoff fixed and exclude training variability and threshold-selection uncertainty. They are not CIs for the difference between two models. **CIs were not recalculated for the unfiltered MedGemma results; the old token-filtered intervals must not be reused.**
- Full-dataset YOLO prediction batch is 8; MedGemma prediction batch is 1. The separate timing benchmark is batch 1 for both.

## 1. YOLO26s ablation: seed 42

**Schedule caveat:** both seed-42 baselines were extended from 50 to a 100-epoch target, restoring optimizer/scaler/EMA state. Their continuation records explicitly say this is not a planned-from-start 100-epoch experiment. HN runs use a direct continuous 100-epoch schedule, so the seed-42 comparison mixes sampling and schedule differences.

| Configuration | AP50:95 | AP50 | Recall | Precision | F1 | FP/image | Source |
|---|---:|---:|---:|---:|---:|---:|---|
| Baseline 640 | 55.14 | 93.20 | 86.61 | 93.45 | 89.90 | 0.0573 | Y42-B640 |
| Baseline 960 | 55.96 | 93.89 | 87.23 | 92.51 | 89.80 | 0.0667 | Y42-B960 |
| HN 640 | 55.42 | 93.72 | 87.13 | 93.08 | 90.01 | 0.0611 | Y42-H640 |
| HN 960 | 56.01 | 93.94 | 87.97 | 93.11 | 90.46 | 0.0615 | Y42-H960 |

## 2. YOLO26s ablation: seed 43

The four seed-43 configurations are the continuous-100 follow-up. This is the schedule-matched ablation, rather than a seed-only replication of the seed-42 baseline schedule. The full seed-43 training logs are not in this checkout; the saved reports identify the corresponding completed epoch-100 backup checkpoints.

| Configuration | AP50:95 | AP50 | Recall | Precision | F1 | FP/image | Source |
|---|---:|---:|---:|---:|---:|---:|---|
| Baseline 640 | 55.47 | 93.73 | 87.20 | 93.34 | 90.17 | 0.0587 | Y43-B640 |
| Baseline 960 | 55.83 | 93.91 | 88.08 | 91.84 | 89.92 | 0.0739 | Y43-B960 |
| HN 640 | 55.68 | 93.69 | 87.09 | 92.68 | 89.80 | 0.0649 | Y43-H640 |
| HN 960 | 56.01 | 94.16 | 89.36 | 91.02 | 90.18 | 0.0832 | Y43-H960 |

**Configuration:** baseline means ordinary sampling. HN means weighted primary negative sampling using the frozen 178-image hard-negative pool, weight 3.0. The 640px and 960px training recipes use batches 35 and 14, respectively, so the resolution comparison also changes training batch size. These tables evaluate each run's `best.pt` saved in its `epoch_100` backup, not necessarily the last epoch's weights.

### Exact validation operating points

| Configuration | Seed 42 cutoff | Seed 43 cutoff |
|---|---:|---:|
| Baseline 640 | 0.419677734375 | 0.38671875 |
| Baseline 960 | 0.3603515625 | 0.32177734375 |
| HN 640 | 0.37744140625 | 0.37109375 |
| HN 960 | 0.373779296875 | 0.29345703125 |

Use the matching completed validation `evaluation.json` via `--protocol` for test evaluation. Do not substitute a rounded table value, another seed's protocol, or another checkpoint.

### What the ablation supports

- **Higher resolution improves baseline AP50:95 in both recorded seeds**, but not operating-point F1: baseline F1 changes 89.90% to 89.80% for seed 42 and 90.17% to 89.92% for seed 43.
- **HN is not a consistent false-positive reduction.** At 640px, HN raises FP/image in both seeds; its F1 change is +0.10 percentage points for seed 42 but -0.37 for seed 43.
- **At 960px, HN improves F1 relative to baseline 960 in both seeds:** +0.67 and +0.26 percentage points. FP/image decreases for seed 42 but increases for seed 43.
- **Operating-point tradeoff:** seed-43 HN960 exceeds baseline640 F1 by about 0.014 percentage points, while FP/image rises from 0.0587 to 0.0832. These are validation-selected operating points, not an equal-threshold or equal-recall comparison; they do not establish statistical significance or clinical superiority.
- Do not pool the two seeds into a mean ± standard deviation that implies identical training schedules. The held-out test below belongs only to the previously evaluated seed-42 checkpoint.

### YOLO validation uncertainty

| Source | AP50:95 95% CI (%) | F1 95% CI (%) |
|---|---|---|
| Y42-B640 | 53.60 to 56.71 | 88.73 to 91.07 |
| Y42-B960 | 54.52 to 57.35 | 88.52 to 90.95 |
| Y42-H640 | 53.88 to 56.94 | 88.83 to 91.06 |
| Y42-H960 | 54.64 to 57.47 | 89.22 to 91.58 |
| Y43-B640 | 53.96 to 56.99 | 89.01 to 91.27 |
| Y43-B960 | 54.39 to 57.29 | 88.69 to 91.04 |
| Y43-H640 | 54.22 to 57.10 | 88.53 to 90.97 |
| Y43-H960 | 54.58 to 57.50 | 88.95 to 91.31 |

These show AP50:95 and F1 intervals; the source `metrics.csv` files also retain the other four metric intervals at full precision. Overlapping marginal intervals are not a paired significance test.

## 3. MedGemma fine-tuning: seed 42

**Evaluated run:** `medgemma_train_2gpu_20260917_134859` (MG42). Base model: `google/medgemma-1.5-4b-it`, revision `91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b`.

The previously reviewed archived run configuration and final recovery manifest record:

- Three completed epochs, 2,520 optimizer steps, seed 42; two GPUs with microbatch 1 per GPU and accumulation 8, giving effective batch 16.
- Language-model-only QLoRA: rank 16, alpha 32, dropout 0.05; NF4 4-bit double quantization with BF16 compute. Base weights and the vision components remain frozen.
- Fused AdamW, learning rate `1e-4`, weight decay 0.01, cosine schedule, 3% warmup, gradient clipping 1.0, SDPA, and non-reentrant gradient checkpointing.
- Full-image RGB input with the saved processor. Targets are fracture-box JSON in normalized 0–1000 coordinates, or `[]` for negatives. Only assistant-answer tokens, including the stop token, contribute to training loss; prompt/image tokens are masked.

These are recorded historical-run settings, **not assumptions from the current configurable launcher defaults**. The archived run-config SHA256 is `fd1b3667fac5b979560ddfe30c1c440567a932b5f44ae6fcdb2674903c8d215c`, also recorded in the local evaluation's `checkpoint_source.run_config_sha256`. The full training configuration/summary is not bundled locally.

The evaluator loaded the saved **best_adapter from checkpoint-2520**, selected by minimum full-validation assistant-token loss, **not localization AP**. Inference was greedy, one beam, maximum 768 new tokens, with a 2,048-token no-truncation guard.

### Updated unfiltered validation results

Recalculated offline with the tracked standard-library-only `rescore_medgemma.py` from `runs/evaluation_medgemma_val_20260918_120322_626065Z/predictions.jsonl`. All **2,050 generated boxes** were present in the saved predictions, alongside **2,734 ground-truth fracture boxes** across **2,895 images**. The collection floor and box cap had discarded no generated boxes.

```bash
python3 -B rescore_medgemma.py \
  --predictions runs/evaluation_medgemma_val_20260918_120322_626065Z/predictions.jsonl \
  --output runs/medgemma_val_unfiltered
```

Use a new output directory each time. The original prediction/report pair is not bundled in Git; README section 5 supplies a pinned-checkpoint validation command to generate a new pair when needed. `rescore.json` records source hashes, code/runtime identity, method, and counts; it is not an inference/test validation protocol. Unit tests cover ambiguous matching, duplicates, order/score independence, retention checks, and the optional archived-result regression.

**Method:** keep every generated box, ignore token scores, and construct prediction-to-ground-truth edges for IoU >= 0.50 within each image. Use maximum-cardinality one-to-one matching (augmenting paths) to count matched fractures; unmatched predictions are FPs and unmatched ground-truth boxes are FNs. IoU arithmetic retains the original evaluator's float32/epsilon convention. No confidence cutoff or new threshold selection is used.

| Metric | Unfiltered estimate |
|---|---:|
| Precision (%) | 66.78 |
| Lesion recall (%) | 50.07 |
| F1 (%) | 57.23 |
| FP/image | 0.2352 |

- Counts: **TP = 1,369; FP = 681; FN = 1,365**.
- Formulas: precision `1369/2050`; recall `1369/2734`; F1 `2738/4784`; FP/image `681/2895`.
- **No new confidence intervals were calculated. MedGemma AP values are omitted.**
- The previous saved metrics used the geometric-mean coordinate-token likelihood proxy with cutoff **0.27922365069389343**. That filtered pipeline retained 2,042 boxes (TP 1,369; FP 673; FN 1,365).
- The offline audit reproduced the previous four metrics exactly. Removing the filter added **8 false positives**; replacing confidence-ordered matching with score-independent matching did not change the aggregate counts on this dataset.
- Predictions SHA256: `632ee5d91c2464561a0bdc7fd6016ef727aa7470ffa61909ab00c6ba3af7ab33`. The audit verified this against the original report and confirmed the source `predictions.jsonl`, `metrics.csv`, and `evaluation.json` remained unchanged. Those original reports still describe the historical token-filtered evaluation.
- YOLO results remain validation-cutoff-selected; the updated MedGemma results are unfiltered. This is not an equal-cutoff comparison.

All 2,895 generated outputs were parse-valid: 1,839 nonempty and 1,056 empty, with no malformed/truncated outputs. Format success is not localization success; these output-status counts are not a confusion matrix.

Only this fine-tuned adapter has a saved MedGemma evaluation. There is no corresponding seed-43, held-out test, or untuned-base result in the available records, so this is not a measured before/after fine-tuning gain or a two-seed MedGemma comparison.

## 4. Completed held-out test: YOLO HN 960, seed 42

Source: Y42-H960-T. The cutoff remains **0.373779296875**, selected on validation. The test report's complete protocol and checkpoint identity match Y42-H960, and its `validation_protocol_sha256` matches that validation report exactly:

`47e470c0ed8e5ef2aeb1224bf0b835bfe956e84d293286a8dba2f71da9e47907`

| Metric | Estimate | 95% patient-bootstrap CI |
|---|---:|---|
| AP50:95 (%) | 56.82 | 55.27 to 58.30 |
| AP50 (%) | 94.71 | 93.59 to 95.73 |
| Lesion recall (%) | 88.88 | 87.22 to 90.43 |
| Precision (%) | 92.92 | 91.60 to 94.13 |
| F1 (%) | 90.85 | 89.62 to 92.08 |
| FP/image | 0.0630 | 0.0523 to 0.0753 |

This is the **only completed test result in the available records**. It is not a seed-43 result and not an average across seeds. The test set has already been evaluated; do not use its errors to tune the model, thresholds, or later experiments, and do not describe it as still untouched or as external clinical validation.

## 5. Recorded hardware measurements

All measurements below were recorded on an **NVIDIA GeForce RTX 5090**, not estimated or measured on the documentation machine. The shared harness cycles over 32 sampled in-memory images, with 10 warmups and 100 timed batch-1 calls, CUDA synchronization, disk I/O excluded, and benchmark cutoff **0.25**, separate from the selected evaluation cutoff. YOLO uses FP16; MedGemma uses NF4/BF16 with FP32 adapter parameters.

| Source | Mean latency (ms) | Median (ms) | p95 (ms) | Peak PyTorch allocated (MiB) |
|---|---:|---:|---:|---:|
| Y42-B640 | 12.26 | 12.34 | 12.95 | 71.26 |
| Y42-B960 | 8.36 | 8.36 | 8.91 | 98.25 |
| Y42-H640 | 12.47 | 12.49 | 13.15 | 71.26 |
| Y42-H960 | 12.46 | 12.48 | 13.13 | 98.25 |
| Y43-B640 | 12.47 | 12.47 | 13.30 | 71.26 |
| Y43-B960 | 14.12 | 14.06 | 15.09 | 98.25 |
| Y43-H640 | 12.35 | 12.42 | 13.14 | 71.26 |
| Y43-H960 | 10.27 | 10.26 | 10.89 | 98.25 |
| MG42 | 3017.57 | 3938.51 | 7121.03 | 7322.59 |
| Y42-H960-T | 12.60 | 12.61 | 13.36 | 98.25 |

YOLO timing includes preprocessing, inference, NMS, and framework overhead; MedGemma includes preprocessing, autoregressive generation, box scoring/parsing, filtering, and framework overhead. MedGemma generated a mean 35.22 tokens per timed call. Memory is warmed evaluator-process **PyTorch peak allocated memory**, including the model, not total device usage or training-memory requirements; reserved memory is separately recorded in the source reports.

These runs are not a controlled matched-load speed experiment. Different observed YOLO timings do not show that HN training makes an otherwise identical architecture faster; do not turn these measurements into deployment guarantees.

## 6. Source records and reproducibility

Each source directory contains the original `metrics.csv`, `evaluation.json`, `predictions.jsonl`, and `subgroups.csv`. The tables preserve the saved report values, rounded only for display. Original records remain unchanged under ignored `runs/`; these directory references are local provenance, not files included in the GitHub checkout.

| ID | Directory relative to `runs/` | Immutable HF checkpoint revision |
|---|---|---|
| Y42-B640 | `eval_baseline_640_val_20260918_164536` | `d104d0498236676d0882942a37ee1c353d80cfda` |
| Y42-B960 | `eval_ablation_960_val_20260918_164825` | `20211a98327a75255a2699ddc82d763a597541c0` |
| Y42-H640 | `eval_ablation_hn640_val_20260918_165152` | `598436c79e1ce1081899ce45dd1ae74773dba8d8` |
| Y42-H960 | `eval_ablation_hn960_val_20260918_180128` | `f909d6feb3ec3022ed1931447d14e28e819ec452` |
| Y43-B640 | `seed_43/eval_baseline640_seed43_direct100_val_20260919_160625` | `fb03eac4aec423ec9ae7766c3dd49c69a5b8a29e` |
| Y43-B960 | `seed_43/eval_baseline960_seed43_direct100_val_20260920_065053` | `01213afe2b4e978585339b3a0d5c6e23b38d538b` |
| Y43-H640 | `seed_43/eval_hn640_seed43_direct100_val_20260919_162604` | `8f1273708d8a99113c97a607ba75ca7b01b9814e` |
| Y43-H960 | `seed_43/eval_hn960_seed43_direct100_val_20260919_163051` | `c67e0025d1371b8e0e1f1827a63dd5fdec4ece07` |
| MG42 | `evaluation_medgemma_val_20260918_120322_626065Z` | `bec122fe4557cf6af3468c6e5853ce0fa66d695c` |
| Y42-H960-T | `eval_hn960_test_20260918_184120` | `f909d6feb3ec3022ed1931447d14e28e819ec452` |

- Frozen data/split repository: [`Crimson-Dawn/grazpedwri-frozen-splits`](https://huggingface.co/datasets/Crimson-Dawn/grazpedwri-frozen-splits) (private dataset repository). At immutable revision `e3aec7e0d16358c283ef850730dd42bdd96b151b`, `LATEST_DATA_BACKUP.json` records completed snapshot `c82f835e73885b53df8284c0ae28cf750a815c2557de554a315bfbc34800d618` and manifest SHA256 `1719f37f442512c3c4fcab8350bedc0f27ff2dcebc082761299912b897747034`, matching these experiments. This is backup-metadata verification, not a fresh archive restore.
- YOLO checkpoint repository: `Crimson-Dawn/grazpedwri-yolo26-checkpoints` (private). Exact `epoch_100/best.pt` paths and checkpoint SHA256 values are in each report's `checkpoint_source`.
- MedGemma checkpoint repository: `Crimson-Dawn/medgemma-fracture-checkpoints` (private). Selected archive: `runs/medgemma_train_2gpu_20260917_134859/backups/final_step_002520_20260917_181358_264121Z.zip`; SHA256 `926051d3e0e154ded6c85e8c3bc4537d409208dfcb291b8de5c5897934677d96`.
- Seed-42 baseline schedule evidence is in the two `runs/baseline_yolo26s_{640,960}_100epochs_seed42_from50_*/continuation_info.json` and `args.yaml` records. Seed-43/HN provenance here is the saved report/checkpoint naming and recorded experiment workflow, not a fresh audit of remote training logs.
- During compilation, all ten prediction-file SHA256 checks passed, and the seed-42 test report was verified against its exact completed validation protocol. Saved source/runtime/checkpoint checks remain mandatory for any future execution.

### Exact checkpoint paths within the private model repositories

For YOLO, pass the path below as `--hf-filename`, together with its revision in
the source table above. For MedGemma, use `--hf-run medgemma_train_2gpu_20260917_134859`;
the resolver verifies and restores the final archive listed here.

| Source | Repository-relative checkpoint/archive path |
|---|---|
| Y42-B640 | `runs/baseline_yolo26s_640_100epochs_seed42_from50_20260917_163013_311680Z/epoch_100/best.pt` |
| Y42-B960 | `runs/baseline_yolo26s_960_100epochs_seed42_from50_20260917_163034_265962Z/epoch_100/best.pt` |
| Y42-H640 | `runs/hardneg_yolo26s_640_100epochs_seed42_20260918_123707_008523Z/epoch_100/best.pt` |
| Y42-H960 | `runs/hardneg_yolo26s_960_100epochs_seed42_20260918_092553_812045Z/epoch_100/best.pt` |
| Y43-B640 | `runs/baseline_yolo26s_640_100epochs_seed43_20260919_024031_450692Z/epoch_100/best.pt` |
| Y43-B960 | `runs/baseline_yolo26s_960_100epochs_seed43_20260919_091740_052280Z/epoch_100/best.pt` |
| Y43-H640 | `runs/hardneg_yolo26s_640_100epochs_seed43_direct100_20260919_041817_261415Z/epoch_100/best.pt` |
| Y43-H960 | `runs/hardneg_yolo26s_960_100epochs_seed43_direct100_20260919_025208_769491Z/epoch_100/best.pt` |
| MG42 | `runs/medgemma_train_2gpu_20260917_134859/backups/final_step_002520_20260917_181358_264121Z.zip` |
| Y42-H960-T | `runs/hardneg_yolo26s_960_100epochs_seed42_20260918_092553_812045Z/epoch_100/best.pt` |

Return to the [workflow README](README.md).
