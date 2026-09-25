"""Shared single-class fracture evaluation. Both model scripts call this file.

Six detection metrics, validation-only threshold selection, patient bootstrap,
subgroups and timing. No model loading or test-time tuning happens here.
IoU/AP matching/integration reproduce Ultralytics 8.4.152 (AGPL-3.0); see LICENSE.
"""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import argparse
import gc
import hashlib
import json
import sys
import time
import warnings

import numpy as np
import pandas as pd

EXPECTED_MANIFEST = "1719f37f442512c3c4fcab8350bedc0f27ff2dcebc082761299912b897747034"
EXPECTED_ULTRALYTICS = "8.4.152"
SCORE_FLOOR, MAX_DETECTIONS, OPERATING_IOU = 0.001, 300, 0.50
BENCHMARK_CONFIDENCE = 0.25
METRICS = ["AP50_95", "AP50", "lesion_recall", "false_positives_per_image", "precision", "F1"]
SELECTION_RULE = "Maximum validation lesion F1 at IoU >= 0.50; exact F1 ties choose the highest tested cutoff"
CI_METHOD = ("95% percentile whole-patient bootstrap; image/lesion-weighted metrics; model and cutoff fixed; "
             "excludes training variability and threshold-selection uncertainty")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def common_arguments(parser, model_name):
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path, help="Local .pt (YOLO), or MedGemma run/adapter/checkpoint directory or backup ZIP")
    source.add_argument("--hf", action="store_true", help="Download the checkpoint from Hugging Face")
    parser.add_argument("--hf-repo", default=None, help="Optional repo override; defaults to your model's checkpoint repo")
    parser.add_argument("--hf-revision", default=None, help="Commit/tag/branch; resolved once to an immutable commit")
    parser.add_argument("--cache-dir", type=Path, default=Path.home() / ".cache/fracture_evaluation")
    parser.add_argument("--dataset", type=Path, default=Path.cwd() / "data/grazpedwri_yolo",
                        help="Frozen prepared YOLO dataset, used by BOTH evaluators for identical images/ground truth")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--protocol", type=Path, help="For test only: this model's completed validation evaluation.json")
    parser.add_argument("--output", type=Path, help="New output directory; never overwrite an existing evaluation")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--bootstraps", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument("--benchmark-runs", type=int, default=100, help="Batch-1 timed runs, following 10 warmups; minimum 10")
    parser.add_argument("--diagnostics", action="store_true", help="Also save threshold sweep, bootstrap draws and evaluated manifest")
    parser.add_argument("--smoke", action="store_true", help="Validation only: check 2 positive + 2 negative images; no benchmark, metrics or usable test protocol")
    parser.set_defaults(model_name=model_name)


def check_arguments(args):
    require(args.bootstraps >= 100, "Use at least 100 patient bootstraps; 1000 recommended.")
    require(args.benchmark_runs >= 10 and args.device >= 0, "Use >=10 benchmark runs and a nonnegative device.")
    require((args.split == "test") == (args.protocol is not None), "Only test requires --protocol from completed validation.")
    require(not args.smoke or args.split == "val", "Smoke checks cannot touch the held-out test split.")
    require(args.hf or (args.hf_repo is None and args.hf_revision is None), "Do not combine local weights with HF options.")


def load_cohort(root, split):
    """Read frozen metadata and ONLY this split's images/labels; never resplit."""
    root = Path(root).resolve()
    manifest_path = root / "split_manifest.csv"
    require(sha256(manifest_path) == EXPECTED_MANIFEST, "The frozen split manifest has changed.")
    info = json.loads((root / "dataset_info.json").read_text())
    require(info["manifest_sha256"] == EXPECTED_MANIFEST, "Dataset provenance differs from the frozen manifest.")
    manifest = pd.read_csv(manifest_path)
    required = {"filestem", "patient_id", "split", "sample_type", "fracture_count", "fracture_labels_json", "image_relpath"}
    require(required.issubset(manifest.columns), "Manifest is missing required evaluation columns.")
    require(manifest.filestem.is_unique and manifest.patient_id.notna().all(), "Missing/duplicate image or patient ID.")
    require(manifest.split.isin(["train", "val", "test"]).all(), "Unknown split.")
    require(manifest.groupby("patient_id")["split"].nunique().eq(1).all(), "Patient leakage across splits.")
    cohort = manifest[manifest.split.eq(split) & manifest.sample_type.isin(["positive", "negative"])].copy().sort_values("filestem")
    require(len(cohort) and cohort.fracture_count.sum() > 0, "Evaluation needs images and labeled fractures.")
    require(cohort.patient_id.nunique() >= 2, "Patient bootstrap needs at least two patients.")
    paths, labels = {}, {}
    for row in cohort.itertuples(index=False):
        require(int(row.filestem.split("_")[0]) == int(row.patient_id), f"Filename/patient mismatch: {row.filestem}")
        image = root / "images" / split / f"{row.filestem}{Path(row.image_relpath).suffix}"
        label = root / "labels" / split / f"{row.filestem}.txt"
        require(image.is_file() and label.is_file(), f"Missing prepared files: {row.filestem}")
        require(image.resolve().stem == row.filestem, f"Image link points at another image: {row.filestem}")
        values = [list(map(float, line.split())) for line in label.read_text().splitlines() if line.strip()]
        boxes = np.asarray(values, dtype=np.float32).reshape(-1, 5)
        expected = np.asarray(json.loads(row.fracture_labels_json), dtype=np.float32).reshape(-1, 5)
        require(boxes.shape == expected.shape and np.allclose(boxes, expected, atol=1e-6, rtol=0),
                f"Prepared labels differ from manifest: {row.filestem}")
        require(len(boxes) == row.fracture_count and bool(len(boxes)) == (row.sample_type == "positive"), "Label count/category mismatch.")
        require(np.isfinite(boxes).all() and (boxes[:, 0] == 0).all() and (boxes[:, 3:] > 0).all(), "Invalid fracture boxes.")
        require(((boxes[:, 1:] >= 0) & (boxes[:, 1:] <= 1)).all(), "Invalid normalized box coordinates.")
        paths[row.filestem], labels[row.filestem] = image, boxes[:, 1:]
    return cohort, paths, labels


def evaluation_cohort(args):
    cohort, paths, labels = load_cohort(args.dataset, args.split)
    if args.smoke:
        cohort = pd.concat([cohort[cohort.sample_type.eq(kind)].head(2)
                            for kind in ("positive", "negative")]).sort_values("filestem")
        paths = {stem: paths[stem] for stem in cohort.filestem}
        labels = {stem: labels[stem] for stem in cohort.filestem}
    return cohort, paths, labels


def finish_smoke(args, output, records, checkpoint, identity):
    if not args.smoke:
        return False
    write_json(output / "evaluation.json", {"status": "smoke_only", "split": "val", "images": len(records),
               "output_status_counts": dict(Counter(r["status"] for r in records)), "identity": identity,
               "checkpoint_source": checkpoint, "note": "No accuracy estimate or frozen test protocol; inspect predictions.jsonl."})
    print("Smoke complete. Inspect predictions.jsonl, then rerun WITHOUT --smoke for full validation.", flush=True)
    return True


def ground_truth_xyxy(normalized_xywh, width, height):
    boxes = np.empty((len(normalized_xywh), 4), dtype=np.float32)
    boxes[:, :2] = (normalized_xywh[:, :2] - normalized_xywh[:, 2:] / 2) * [width, height]
    boxes[:, 2:] = (normalized_xywh[:, :2] + normalized_xywh[:, 2:] / 2) * [width, height]
    return boxes


# These two routines retain the pinned Ultralytics numerical operations and matching
# order. Keeping this small single-class subset avoids installing YOLO in the MedGemma
# environment. Regression tests compare it to the actual 8.4.152 source.
def pairwise_iou(gt, pred):
    import torch
    a, b = torch.from_numpy(gt).float(), torch.from_numpy(pred).float()
    (a1, a2), (b1, b2) = a.unsqueeze(1).chunk(2, 2), b.unsqueeze(0).chunk(2, 2)
    intersection = (torch.min(a2, b2) - torch.max(a1, b1)).clamp_(0).prod(2)
    return (intersection / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - intersection + 1e-7)).numpy()


def ap_matches(ious):
    import torch
    correct = np.zeros((ious.shape[1], 10), dtype=bool)
    for column, threshold in enumerate(torch.linspace(0.5, 0.95, 10).tolist()):
        matches = np.array(np.nonzero(ious >= threshold)).T
        if len(matches):
            if len(matches) > 1:
                matches = matches[ious[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), column] = True
    return correct


def operating_matches(ious):
    """Greedy confidence-ordered, one-to-one matching at IoU >= .50."""
    n_gt, n_pred = ious.shape
    correct, used = np.zeros(n_pred, dtype=bool), np.zeros(n_gt, dtype=bool)
    if n_gt:
        for j in range(n_pred):
            overlaps = ious[:, j].copy()
            overlaps[used] = -1
            k = int(overlaps.argmax())
            if overlaps[k] >= OPERATING_IOU:
                correct[j], used[k] = True, True
    return correct


def average_precision(correct, scores, n_gt):
    """Exact single-class AP subset of ap_per_class/compute_ap in 8.4.152."""
    tp = correct[np.argsort(-scores)]  # Keep Ultralytics' tie behavior, not a new AP definition.
    tp_sum, fp_sum = tp.cumsum(0), (1 - tp).cumsum(0)
    recall = tp_sum / (n_gt + 1e-16)
    precision = tp_sum / (tp_sum + fp_sum)
    result = []
    for column in range(10):
        r, p = recall[:, column], precision[:, column]
        mrec = np.concatenate(([0.0], r, [r[-1] if len(r) else 1.0], [1.0]))
        mpre = np.concatenate(([1.0], p, [0.0], [0.0]))
        mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
        x = np.linspace(0, 1, 101)
        integrate = getattr(np, "trapezoid", None) or np.trapz
        result.append(float(integrate(np.interp(x, mrec, mpre), x)))
    return result[0], float(np.mean(result))


def make_record(stem, patient, gt, pred, scores, status="valid_nonempty"):
    """The ONLY path from either model's boxes into detection metrics."""
    gt = np.asarray(gt, dtype=np.float32).reshape(-1, 4)
    pred = np.asarray(pred, dtype=np.float32).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float32)
    require(scores.ndim == 1 and len(pred) == len(scores), "Score/box count mismatch.")
    require(np.isfinite(pred).all() and np.isfinite(scores).all(), "Nonfinite prediction.")
    require(((scores >= 0) & (scores <= 1)).all(), "Prediction scores must lie in [0,1].")
    require((pred[:, 2:] >= pred[:, :2]).all(), "Reversed predicted box.")  # Clipped zero-area YOLO boxes remain FPs, as before.
    order = np.argsort(-scores, kind="stable")
    order = order[scores[order] > SCORE_FLOOR]
    capped = len(order) >= MAX_DETECTIONS
    order = order[:MAX_DETECTIONS]
    pred, scores = pred[order], scores[order]
    overlaps = pairwise_iou(gt, pred)
    return {"stem": stem, "patient": int(patient), "n_gt": len(gt), "scores": scores,
            "ap_correct": ap_matches(overlaps), "op_correct": operating_matches(overlaps),
            "gt_xyxy": gt, "pred_xyxy": pred, "status": status, "at_max_det": capped}


def choose_threshold(records):
    scores = np.concatenate([r["scores"] for r in records])
    correct = np.concatenate([r["op_correct"] for r in records])
    n_gt = sum(r["n_gt"] for r in records)
    require(n_gt > 0, "Cannot select a threshold without fractures.")
    if not len(scores):
        warnings.warn("No predictions: cutoff=1.0, all operating metrics zero except FP/image.")
        return 1.0, pd.DataFrame(columns=["confidence", "precision", "recall", "F1"])
    order = np.argsort(-scores, kind="stable")
    scores, correct = scores[order], correct[order]
    ends = np.flatnonzero(np.r_[scores[:-1] != scores[1:], True])
    tp, n_pred = correct.cumsum()[ends], ends + 1
    cutoffs = [float(scores[end + 1]) if end + 1 < len(scores) else SCORE_FLOOR for end in ends]
    table = pd.DataFrame({"confidence": np.r_[1.0, cutoffs], "precision": np.r_[0.0, tp / n_pred],
                          "recall": np.r_[0.0, tp / n_gt], "F1": np.r_[0.0, 2 * tp / (n_pred + n_gt)]})
    best = int(np.argmax(table.F1.to_numpy()))
    if table.F1.iloc[best] == 0:
        warnings.warn("No correct detections at IoU .50; selected validation F1 is zero.")
    return float(table.confidence.iloc[best]), table


def summarize(records, threshold):
    n_images, n_gt = len(records), sum(r["n_gt"] for r in records)
    require(n_images > 0, "Cannot summarize an empty cohort.")
    scores = np.concatenate([r["scores"] for r in records])
    ap_correct = np.concatenate([r["ap_correct"] for r in records], axis=0)
    op_correct = np.concatenate([r["op_correct"] for r in records])
    keep = scores > threshold
    n_pred, tp = int(keep.sum()), int(op_correct[keep].sum())
    if not n_gt:
        ap50 = ap5095 = np.nan
    elif not len(scores) or not ap_correct.any():
        ap50 = ap5095 = 0.0
    else:
        ap50, ap5095 = average_precision(ap_correct, scores, n_gt)
    return {"AP50_95": ap5095, "AP50": ap50, "lesion_recall": tp / n_gt if n_gt else np.nan,
            "false_positives_per_image": (n_pred - tp) / n_images, "precision": tp / n_pred if n_pred else 0.0,
            "F1": 2 * tp / (n_pred + n_gt) if n_pred + n_gt else np.nan}


def bootstrap(records, threshold, count, seed):
    by_patient = {}
    for record in records:
        by_patient.setdefault(record["patient"], []).append(record)
    ids = np.array(sorted(by_patient))
    rng, draws = np.random.default_rng(seed), []
    for index in range(count):
        sampled = rng.choice(ids, size=len(ids), replace=True)
        draws.append(summarize([r for pid in sampled for r in by_patient[pid]], threshold))
        if (index + 1) % 100 == 0:
            print(f"Patient bootstrap {index + 1}/{count}", flush=True)
    return pd.DataFrame(draws)


def subgroups(cohort, records, threshold):
    groups, rows = {}, []
    if "projection" in cohort:
        groups["view"] = cohort.projection.astype("string").str.strip().replace("", pd.NA).fillna("unknown")
    if "age" in cohort:
        groups["age_years"] = pd.cut(pd.to_numeric(cohort.age, errors="coerce"), [0, 6, 11, 16, np.inf],
                                     right=False, labels=["0-<6", "6-<11", "11-<16", "16+"]).astype(object).fillna("unknown")
    if "gender" in cohort:
        groups["gender"] = cohort.gender.fillna("unknown").astype(str)
    for name in ["cast", "diagnosis_uncertain"]:
        if name in cohort:
            groups[name] = pd.to_numeric(cohort[name], errors="coerce").map({1: "flagged", 0: "not_flagged"}).fillna("unknown")
    indexed = {r["stem"]: r for r in records}
    for family, labels in groups.items():
        for group, frame in cohort.assign(_group=labels).groupby("_group", sort=False):
            subset = [indexed[stem] for stem in frame.filestem]
            rows.append({"subgroup_type": family, "subgroup": str(group), "patients": int(frame.patient_id.nunique()),
                         "images": len(frame), "positive_images": int(frame.sample_type.eq("positive").sum()),
                         "negative_images": int(frame.sample_type.eq("negative").sum()),
                         "fracture_boxes": int(frame.fracture_count.sum()), **summarize(subset, threshold)})
    return pd.DataFrame(rows)


def protocol_identity(model, checkpoint, settings, source_files):
    """Paths/remote location do not affect identity: the same downloaded weights may be used locally."""
    import torch
    identity_keys = ["weights_sha256", "config_sha256", "base_model", "base_revision", "prompt_sha256", "processor_sha256"]
    return {"schema": 1, "model": model, "checkpoint": {k: checkpoint[k] for k in identity_keys if k in checkpoint},
            "raw_manifest_sha256": EXPECTED_MANIFEST, "inference": settings,
            "metric_engine": "Ultralytics-8.4.152-single-class-subset-v1", "operating_iou": OPERATING_IOU,
            "score_floor": SCORE_FLOOR, "max_detections": MAX_DETECTIONS, "confidence_comparison": "score > cutoff",
            "selection_rule": SELECTION_RULE, "benchmark_confidence": BENCHMARK_CONFIDENCE,
            "tie_policy": "Sorted filestems; stable original box order within image score ties; pinned Ultralytics AP sort/match",
            "versions": {"torch": str(torch.__version__), "numpy": np.__version__, "pandas": pd.__version__},
            "source_sha256": {Path(p).name: sha256(p) for p in source_files}}


def frozen_protocol(args, identity):
    if args.split == "val":
        return None
    previous = json.loads(args.protocol.read_text())
    require(previous.get("status") == "complete" and previous.get("split") == "val",
            "Test requires a COMPLETE validation evaluation.json from these scripts (not a legacy operating_point.json).")
    protocol = previous["protocol"]
    require(protocol["identity"] == identity, "Model, data, source, software or inference settings changed. Reevaluate validation first.")
    require(protocol["selected_on"] == "val", "Threshold was not selected on validation.")
    threshold = protocol["confidence_cutoff"]
    require(isinstance(threshold, (int, float)) and np.isfinite(threshold) and SCORE_FLOOR <= threshold <= 1,
            "Invalid frozen confidence cutoff.")
    return protocol


def create_output(args):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%fZ")
    output = args.output or Path.cwd() / "runs" / f"evaluation_{args.model_name}_{args.split}_{stamp}"
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    print("Output:", output, flush=True)
    return output


def save_prediction(handle, record, width, height, image_path, details=None):
    value = {"filestem": record["stem"], "patient_id": record["patient"], "width": width, "height": height,
             "gt_xyxy": record["gt_xyxy"].tolist(), "pred_xyxy": record["pred_xyxy"].tolist(),
             "scores": record["scores"].tolist(), "status": record["status"], "at_max_det": record["at_max_det"],
             "image_sha256": sha256(image_path)}
    value.update(details or {})
    handle.write(json.dumps(value, allow_nan=False) + "\n")
    handle.flush()


def benchmark(predict, paths, load_image, device, runs, precision, scope, forward_phase=False):
    """Batch 1, cached inputs, CUDA synchronization, 10 warmups; same harness for both models."""
    import torch
    rng = np.random.default_rng(1729)
    chosen = rng.choice(len(paths), size=min(32, len(paths)), replace=False)
    images = [load_image(paths[i]) for i in chosen]
    require(all(image is not None for image in images), "Benchmark image load failed.")
    elapsed, forward, tokens = [], [], []
    try:
        torch.cuda.synchronize(device)
        gc.collect()
        torch.cuda.empty_cache()
        for i in range(10):
            result = predict(images[i % len(images)], BENCHMARK_CONFIDENCE)
            del result
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        for i in range(runs):
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            result = predict(images[i % len(images)], BENCHMARK_CONFIDENCE)
            torch.cuda.synchronize(device)
            elapsed.append((time.perf_counter() - started) * 1000)
            if forward_phase:
                forward.append(result[0].speed["inference"])
            elif "generated_tokens" in result:
                tokens.append(result["generated_tokens"])
            del result
            if (i + 1) % 10 == 0:
                print(f"Batch-1 benchmark {i + 1}/{runs}", flush=True)
        return {"gpu": torch.cuda.get_device_name(device), "batch_size": 1, "precision": precision,
                "confidence_cutoff": BENCHMARK_CONFIDENCE, "warmup_runs": 10, "timed_runs": runs,
                "timing_scope": scope + "; in-memory input, disk I/O excluded, CUDA synchronized",
                "latency_mean_ms": float(np.mean(elapsed)), "latency_median_ms": float(np.median(elapsed)),
                "latency_p95_ms": float(np.percentile(elapsed, 95)),
                "model_forward_mean_ms": float(np.mean(forward)) if forward else None,
                "generated_tokens_mean": float(np.mean(tokens)) if tokens else None,
                "peak_torch_allocated_MiB": torch.cuda.max_memory_allocated(device) / 1024**2,
                "peak_torch_reserved_MiB": torch.cuda.max_memory_reserved(device) / 1024**2,
                "memory_scope": "Evaluator process PyTorch peak during warmed batch-1 inference, including model; not total device memory"}
    finally:
        for image in images:
            if hasattr(image, "close"):
                image.close()


def finish(args, output, cohort, records, checkpoint, identity, frozen, bench):
    """Write four default files. Protocol+benchmark+provenance share evaluation.json."""
    require([r["stem"] for r in records] == cohort.filestem.tolist(), "Missing, duplicate or reordered evaluation images.")
    if frozen is None:
        threshold, sweep = choose_threshold(records)
        protocol = {"selected_on": "val", "confidence_cutoff": threshold, "identity": identity}
    else:
        threshold, sweep, protocol = frozen["confidence_cutoff"], None, frozen
    point = summarize(records, threshold)
    if sweep is not None and len(sweep):
        require(np.isclose(point["F1"], sweep.F1.max()), "Threshold sweep mismatch.")
    draws = bootstrap(records, threshold, args.bootstraps, args.bootstrap_seed)
    metrics = pd.DataFrame([{"metric": key, "estimate": point[key], "CI95_low": float(draws[key].quantile(.025)),
                             "CI95_high": float(draws[key].quantile(.975)),
                             "valid_bootstrap_replicates": int(draws[key].notna().sum())} for key in METRICS])
    metrics.to_csv(output / "metrics.csv", index=False)
    groups = subgroups(cohort, records, threshold)
    groups.to_csv(output / "subgroups.csv", index=False)
    statuses = Counter(r["status"] for r in records)
    negative_statuses = Counter(r["status"] for r in records if r["n_gt"] == 0)
    write_json(output / "evaluation.json", {
        "status": "complete", "split": args.split, "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model_name, "protocol": protocol, "checkpoint_source": checkpoint,
        "counts": {"images": len(records), "patients": int(cohort.patient_id.nunique()),
                   "fracture_boxes": sum(r["n_gt"] for r in records), "positive_images": int(cohort.sample_type.eq("positive").sum()),
                   "negative_images": int(cohort.sample_type.eq("negative").sum()),
                   "images_at_max_det": sum(int(r["at_max_det"]) for r in records)},
        "output_status_counts": dict(statuses), "negative_image_output_status_counts": dict(negative_statuses),
        "failure_policy": "Malformed/truncated model output gives zero predictions but NEVER removes an image or its GT; it is not a valid negative. Infrastructure/scoring failures abort.",
        "benchmark": bench, "bootstrap": {"count": args.bootstraps, "seed": args.bootstrap_seed, "method": CI_METHOD},
        "subgroups": "Counts and point estimates, no CIs; raw view codes; missing metadata unknown; patients may occur in multiple rows",
        "metric_units": "AP, recall, precision and F1 are fractions; false positives per image is a count rate",
        "predictions_sha256": sha256(output / "predictions.jsonl"),
        "validation_protocol_sha256": sha256(args.protocol) if args.protocol else None, "python": sys.version,
    })
    if args.diagnostics:
        draws.to_csv(output / "bootstrap_replicates.csv", index=False)
        cohort.to_csv(output / "evaluated_manifest.csv", index=False)
        if sweep is not None:
            sweep.to_csv(output / "validation_threshold_sweep.csv", index=False)
    if (metrics.valid_bootstrap_replicates < args.bootstraps).any():
        warnings.warn("Some bootstrap draws had no fractures; undefined AP/recall excluded from CIs. See valid replicate counts.")
    if any(r["at_max_det"] for r in records):
        warnings.warn("Some images hit max_det=300; inspect evaluation.json.")
    print(f"\nFrozen confidence cutoff: {threshold:.8f}\n{metrics.to_string(index=False)}", flush=True)
    print("\nSaved:", output, flush=True)
    print("Use this validation evaluation.json unchanged with --protocol for the final test run.", flush=True)
