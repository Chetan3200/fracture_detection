#!/usr/bin/env python3
"""Offline, standard-library-only rescore of saved MedGemma validation predictions.

This creates a separate unfiltered maximum-cardinality IoU=.50 summary.  It
never invokes inference, changes saved predictions/evaluations, uses scores to
filter boxes, or produces a validation protocol report.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import re
import struct
import sys
from pathlib import Path

from config import MANIFEST_SHA256

IOU_THRESHOLD = 0.50
EPSILON = 1e-7
SCHEMA_VERSION = 1
METHOD_VERSION = "medgemma_unfiltered_maximum_matching_v1"
METRIC_NAMES = ("precision", "lesion_recall", "F1", "false_positives_per_image")
_FENCE = re.compile(r"\A\s*Final Answer:\s*```json\s*(.*?)\s*```\s*\Z", re.S)


def fail(message):
    raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def float32(value):
    """Match the saved evaluator's torch float32 arithmetic without numpy."""
    return struct.unpack("!f", struct.pack("!f", float(value)))[0]


def iou_float32(a, b):
    a, b = [float32(v) for v in a], [float32(v) for v in b]
    width = max(0.0, float32(min(a[2], b[2]) - max(a[0], b[0])))
    height = max(0.0, float32(min(a[3], b[3]) - max(a[1], b[1])))
    intersection = float32(width * height)
    area_a = float32(float32(a[2] - a[0]) * float32(a[3] - a[1]))
    area_b = float32(float32(b[2] - b[0]) * float32(b[3] - b[1]))
    return float32(intersection / float32(float32(float32(area_a + area_b) - intersection) + float32(EPSILON)))


def maximum_matching(edges, n_ground_truth):
    """Kuhn augmenting paths: maximum-cardinality, one-to-one matching."""
    owner = [-1] * n_ground_truth

    def augment(prediction, seen):
        for ground_truth in edges[prediction]:
            if seen[ground_truth]:
                continue
            seen[ground_truth] = True
            if owner[ground_truth] == -1 or augment(owner[ground_truth], seen):
                owner[ground_truth] = prediction
                return True
        return False

    return sum(augment(index, [False] * n_ground_truth) for index in range(len(edges)))


def canonical_predictions(boxes):
    """Coordinate order intentionally removes saved confidence ordering."""
    return sorted((list(box) for box in boxes), key=lambda box: tuple(box))


def matched_count(boxes, ground_truth):
    ordered = canonical_predictions(boxes)
    edges = []
    for box in ordered:
        candidates = [(iou_float32(gt, box), index) for index, gt in enumerate(ground_truth)]
        # IoU order is deterministic only; scores are not inspected.
        edges.append([index for overlap, index in sorted(candidates, key=lambda item: (-item[0], item[1]))
                      if overlap >= IOU_THRESHOLD])
    return maximum_matching(edges, len(ground_truth))


def _unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            fail("generated_text has a duplicate JSON key")
        result[key] = value
    return result


def _reject_json_constant(value):
    fail("generated_text has a nonfinite JSON constant")


def _raw_boxes(record):
    text = record.get("generated_text")
    if not isinstance(text, str):
        fail("generated_text must be a string")
    match = _FENCE.fullmatch(text)
    if not match:
        fail("generated_text is not a complete Final Answer JSON fence")
    try:
        raw = json.loads(match.group(1), object_pairs_hook=_unique_json_object,
                         parse_constant=_reject_json_constant)
    except json.JSONDecodeError as exc:
        fail("generated_text has invalid JSON: %s" % exc.msg)
    if not isinstance(raw, list):
        fail("generated_text JSON must be a list")
    boxes = []
    for item in raw:
        if not isinstance(item, dict) or set(item) != {"label", "box_2d"} or item["label"] != "fracture":
            fail("generated_text has an invalid MedGemma box object")
        box = item["box_2d"]
        if not isinstance(box, list) or len(box) != 4 or not all(type(x) in (int, float) and math.isfinite(x) for x in box):
            fail("generated_text has an invalid box_2d")
        ymin, xmin, ymax, xmax = box
        if not (0 <= ymin < ymax <= 1000 and 0 <= xmin < xmax <= 1000):
            fail("generated_text has an out-of-range or degenerate box_2d")
        boxes.append(box)
    return boxes


def _validate_boxes(boxes, name):
    if not isinstance(boxes, list):
        fail("%s must be a list" % name)
    for box in boxes:
        if not isinstance(box, list) or len(box) != 4 or not all(type(value) in (int, float) and math.isfinite(value) for value in box):
            fail("%s contains an invalid box" % name)


def _validate_ground_truth_boxes(boxes, width, height):
    _validate_boxes(boxes, "gt_xyxy")
    for x1, y1, x2, y2 in boxes:
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            fail("gt_xyxy contains an out-of-image, reversed, or degenerate box")


def validate_record(record):
    required = ("filestem", "width", "height", "gt_xyxy", "pred_xyxy", "scores", "status",
                "truncated", "at_max_det", "generated_text", "emitted_box_count_before_collection")
    if not isinstance(record, dict) or any(key not in record for key in required):
        fail("prediction record is missing required fields")
    if not isinstance(record["filestem"], str) or not record["filestem"]:
        fail("prediction record has invalid filestem")
    if type(record["width"]) is not int or type(record["height"]) is not int or record["width"] <= 0 or record["height"] <= 0:
        fail("prediction record has invalid image dimensions")
    if record["status"] not in ("valid_empty", "valid_nonempty"):
        fail("prediction record status must be valid_empty or valid_nonempty")
    if record["truncated"] is not False or record["at_max_det"] is not False:
        fail("prediction record is truncated or capped")
    _validate_ground_truth_boxes(record["gt_xyxy"], record["width"], record["height"])
    _validate_boxes(record["pred_xyxy"], "pred_xyxy")
    if not isinstance(record["scores"], list) or len(record["scores"]) != len(record["pred_xyxy"]):
        fail("prediction score/box count mismatch")
    raw = _raw_boxes(record)
    if type(record["emitted_box_count_before_collection"]) is not int or record["emitted_box_count_before_collection"] != len(raw):
        fail("emitted_box_count_before_collection does not match raw generated_text")
    if len(raw) != len(record["pred_xyxy"]):
        fail("not every raw generated box was retained in pred_xyxy")
    if (record["status"] == "valid_empty") != (len(raw) == 0):
        fail("status does not match raw generated box count")
    # Reconstruct the scorer's image-coordinate conversion and compare float32 values.
    reconstructed = [[float32(xmin * record["width"] / 1000), float32(ymin * record["height"] / 1000),
                      float32(xmax * record["width"] / 1000), float32(ymax * record["height"] / 1000)]
                     for ymin, xmin, ymax, xmax in raw]
    # make_record may reorder retained boxes by saved score.  Compare canonical
    # float32 coordinates so this audit proves retention without treating that
    # saved order as a matching or filtering signal.
    expected_boxes = sorted(reconstructed, key=tuple)
    saved_boxes = sorted(([float32(value) for value in box] for box in record["pred_xyxy"]), key=tuple)
    for expected, saved in zip(expected_boxes, saved_boxes):
        if expected != saved:
            fail("pred_xyxy does not retain raw generated boxes")


def load_predictions(predictions):
    rows, stems = [], set()
    with Path(predictions).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                fail("predictions JSONL has an empty line at %d" % number)
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                fail("predictions JSONL line %d is invalid: %s" % (number, exc.msg))
            validate_record(row)
            if row["filestem"] in stems:
                fail("predictions JSONL has duplicate filestem: %s" % row["filestem"])
            stems.add(row["filestem"])
            rows.append(row)
    if not rows:
        fail("predictions JSONL is empty")
    return rows


def load_source_evaluation(path, predictions_hash):
    try:
        report = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail("cannot read source evaluation: %s" % exc)
    if not isinstance(report, dict) or report.get("status") != "complete" or report.get("model") != "medgemma" or report.get("split") != "val":
        fail("source evaluation must be a complete MedGemma val report")
    if report.get("predictions_sha256") != predictions_hash:
        fail("source evaluation predictions_sha256 does not match --predictions")
    counts = report.get("counts")
    if not isinstance(counts, dict) or type(counts.get("images")) is not int or counts["images"] <= 0:
        fail("source evaluation counts.images must be a positive integer")
    if type(counts.get("fracture_boxes")) is not int or counts["fracture_boxes"] < 0:
        fail("source evaluation counts.fracture_boxes must be a nonnegative integer")
    protocol = report.get("protocol", {})
    identity = protocol.get("identity", {}) if isinstance(protocol, dict) else {}
    if (not isinstance(protocol, dict) or protocol.get("selected_on") != "val"
            or not isinstance(identity, dict) or identity.get("model") != "medgemma"
            or identity.get("raw_manifest_sha256") != MANIFEST_SHA256):
        fail("source evaluation protocol must identify MedGemma and the frozen validation split")
    return report


def _outside(directory, forbidden_parent, message):
    try:
        directory.relative_to(forbidden_parent)
    except ValueError:
        return
    fail(message)


def rescore(predictions, source_evaluation, output):
    output_raw = Path(os.path.abspath(os.fspath(output)))
    if os.path.lexists(os.fspath(output_raw)):
        fail("output path already exists (including a broken symlink)")
    predictions = Path(predictions).resolve()
    source_evaluation = Path(source_evaluation).resolve()
    output = output_raw.resolve()
    _outside(output, source_evaluation.parent, "output directory must not be inside the source evaluation folder")
    _outside(output, predictions.parent, "output directory must not be inside the predictions folder")
    predictions_hash_before = sha256(predictions)
    source_hash_before = sha256(source_evaluation)
    report = load_source_evaluation(source_evaluation, predictions_hash_before)
    rows = load_predictions(predictions)
    images = len(rows)
    ground_truth = sum(len(row["gt_xyxy"]) for row in rows)
    report_counts = report["counts"]
    if report_counts["images"] != images or report_counts["fracture_boxes"] != ground_truth:
        fail("source evaluation counts do not match loaded predictions")
    predictions_count = sum(len(row["pred_xyxy"]) for row in rows)
    true_positives = sum(matched_count(row["pred_xyxy"], row["gt_xyxy"]) for row in rows)
    false_positives, false_negatives = predictions_count - true_positives, ground_truth - true_positives
    metrics = {"precision": true_positives / predictions_count if predictions_count else 0.0,
               "lesion_recall": true_positives / ground_truth if ground_truth else 0.0,
               "F1": 2 * true_positives / (predictions_count + ground_truth) if predictions_count + ground_truth else 0.0,
               "false_positives_per_image": false_positives / images}
    if sha256(predictions) != predictions_hash_before or sha256(source_evaluation) != source_hash_before:
        fail("source predictions or evaluation changed while rescoring")
    payload = {"schema_version": SCHEMA_VERSION, "method_version": METHOD_VERSION,
               "kind": "offline_unfiltered_medgemma_rescore", "not_a_protocol_validation_report": True,
               "rescore_source_sha256": sha256(Path(__file__).resolve()),
               "python": {"executable": sys.executable, "version": sys.version},
               "source": {"predictions": str(predictions), "evaluation": str(source_evaluation),
                          "predictions_sha256": predictions_hash_before, "evaluation_sha256": source_hash_before,
                          "frozen_manifest_sha256": MANIFEST_SHA256},
               "counts": {"images": images, "unique_filestems": len({row['filestem'] for row in rows}),
                          "ground_truth_boxes": ground_truth, "generated_boxes": predictions_count,
                          "TP": true_positives, "FP": false_positives, "FN": false_negatives},
               "method": {"iou_threshold": IOU_THRESHOLD, "float32_epsilon": EPSILON,
                          "iou_arithmetic": "float32 with epsilon 1e-7",
                          "matching": "canonical-coordinate-order maximum-cardinality one-to-one augmenting paths",
                          "boxes": "all retained raw generated boxes; no score filtering, cap, truncation, or score-based matching"},
               "metrics": metrics, "confidence_intervals": None}
    output.mkdir(parents=True)
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("metric", "value"))
        for name in METRIC_NAMES:
            writer.writerow((name, repr(metrics[name])))
    (output / "rescore.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return payload


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True, help="Saved MedGemma predictions.jsonl")
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    parser.add_argument("--source-evaluation", type=Path, help="Complete matching MedGemma val evaluation.json")
    return parser.parse_args(argv)


def main(argv=None):
    args = arguments(argv)
    source = args.source_evaluation if args.source_evaluation is not None else args.predictions.resolve().parent / "evaluation.json"
    try:
        payload = rescore(args.predictions, source, args.output)
    except (OSError, ValueError) as exc:
        raise SystemExit("rescore_medgemma: %s" % exc)
    print(json.dumps({"output": str(args.output.resolve()), "metrics": payload["metrics"]}, allow_nan=False))
    return 0


if __name__ == "__main__":
    main()
