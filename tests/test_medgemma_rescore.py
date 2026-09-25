"""Stdlib-only tests for the offline MedGemma unfiltered rescore utility."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import rescore_medgemma as rescore


def raw_text(boxes):
    return "Final Answer: ```json\n%s\n```" % json.dumps(
        [{"box_2d": box, "label": "fracture"} for box in boxes], separators=(",", ":"))


def saved_box(raw, width=1000, height=1000):
    ymin, xmin, ymax, xmax = raw
    return [rescore.float32(xmin * width / 1000), rescore.float32(ymin * height / 1000),
            rescore.float32(xmax * width / 1000), rescore.float32(ymax * height / 1000)]


def row(stem, raw=(), gt=(), scores=None):
    raw = list(raw)
    return {"filestem": stem, "width": 1000, "height": 1000, "gt_xyxy": [list(x) for x in gt],
            "pred_xyxy": [saved_box(x) for x in raw], "scores": list(scores if scores is not None else [0.1] * len(raw)),
            "status": "valid_empty" if not raw else "valid_nonempty", "truncated": False, "at_max_det": False,
            "generated_text": raw_text(raw), "emitted_box_count_before_collection": len(raw)}


class MedGemmaRescoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def write_inputs(self, rows, report=True):
        source = self.root / "source"
        source.mkdir(exist_ok=True)
        predictions = source / "predictions.jsonl"
        predictions.write_text("".join(json.dumps(item) + "\n" for item in rows), encoding="utf-8")
        evaluation = source / "evaluation.json"
        if report:
            evaluation.write_text(json.dumps({
                "status": "complete", "model": "medgemma", "split": "val",
                "predictions_sha256": hashlib.sha256(predictions.read_bytes()).hexdigest(),
                "counts": {"images": len(rows), "fracture_boxes": sum(len(item["gt_xyxy"]) for item in rows)},
                "protocol": {"selected_on": "val", "identity": {
                    "model": "medgemma", "raw_manifest_sha256": rescore.MANIFEST_SHA256}},
            }), encoding="utf-8")
        return predictions, evaluation

    def test_rejects_wrong_or_missing_frozen_protocol(self):
        for protocol in ({}, {"selected_on": "val", "identity": {
                "model": "medgemma", "raw_manifest_sha256": "0" * 64}},
                {"selected_on": "val", "identity": {
                    "model": "yolo26", "raw_manifest_sha256": rescore.MANIFEST_SHA256}}):
            predictions, evaluation = self.write_inputs([row("one")])
            report = json.loads(evaluation.read_text())
            report["protocol"] = protocol
            evaluation.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "frozen validation split"):
                rescore.rescore(predictions, evaluation, self.root / "bad-protocol")

    def test_toy_conflict_requires_augmenting_path(self):
        self.assertEqual(rescore.maximum_matching([[0, 1], [0]], 2), 2)
        self.assertEqual(rescore.maximum_matching([[0], [0]], 1), 1)

    def test_duplicate_predictions_count_one_true_positive(self):
        gt = [[0, 0, 100, 100]]
        self.assertEqual(rescore.matched_count([[0, 0, 100, 100], [0, 0, 100, 100]], gt), 1)

    def test_empty_negative_is_an_image_and_has_no_prediction(self):
        predictions, evaluation = self.write_inputs([row("negative")])
        result = rescore.rescore(predictions, evaluation, self.root / "out")
        self.assertEqual(result["counts"], {"images": 1, "unique_filestems": 1, "ground_truth_boxes": 0,
                                             "generated_boxes": 0, "TP": 0, "FP": 0, "FN": 0})
        self.assertEqual(result["metrics"]["false_positives_per_image"], 0.0)

    def test_rejects_duplicate_hash_mismatch_incomplete_and_missing_raw_boxes(self):
        first = row("same")
        predictions, evaluation = self.write_inputs([first, first])
        with self.assertRaisesRegex(ValueError, "duplicate filestem"):
            rescore.rescore(predictions, evaluation, self.root / "duplicate")
        predictions, evaluation = self.write_inputs([row("one")])
        evaluation.write_text(json.dumps({"status": "complete", "model": "medgemma", "split": "val",
                                           "predictions_sha256": "0" * 64,
                                           "counts": {"images": 1, "fracture_boxes": 0}}))
        with self.assertRaisesRegex(ValueError, "predictions_sha256"):
            rescore.rescore(predictions, evaluation, self.root / "bad-hash")
        predictions, evaluation = self.write_inputs([row("one")])
        evaluation.write_text(json.dumps({"status": "smoke_only", "model": "medgemma", "split": "val",
                                           "predictions_sha256": hashlib.sha256(predictions.read_bytes()).hexdigest(),
                                           "counts": {"images": 1, "fracture_boxes": 0}}))
        with self.assertRaisesRegex(ValueError, "complete MedGemma val"):
            rescore.rescore(predictions, evaluation, self.root / "incomplete")
        broken = row("raw", raw=[[10, 20, 30, 40]])
        broken["pred_xyxy"] = []
        broken["scores"] = []
        predictions, evaluation = self.write_inputs([broken])
        with self.assertRaisesRegex(ValueError, "not every raw generated box"):
            rescore.rescore(predictions, evaluation, self.root / "missing")

    def test_refuses_existing_and_source_folder_output(self):
        predictions, evaluation = self.write_inputs([row("one")])
        existing = self.root / "existing"
        existing.mkdir()
        with self.assertRaisesRegex(ValueError, "already exists"):
            rescore.rescore(predictions, evaluation, existing)
        with self.assertRaisesRegex(ValueError, "inside the source evaluation"):
            rescore.rescore(predictions, evaluation, predictions.parent / "nested")

    def test_ordering_and_scores_do_not_change_matching(self):
        gt = [[0, 0, 100, 100], [100, 0, 200, 100]]
        boxes = [[0, 0, 100, 100], [100, 0, 200, 100]]
        self.assertEqual(rescore.matched_count(boxes, gt), rescore.matched_count(list(reversed(boxes)), gt))
        rows_a = [row("x", raw=[[0, 0, 100, 100], [0, 100, 100, 200]], gt=gt, scores=[0.99, 0.01])]
        rows_b = [row("x", raw=[[0, 100, 100, 200], [0, 0, 100, 100]], gt=gt, scores=[0.0001, 1.0])]
        pa, ea = self.write_inputs(rows_a)
        result_a = rescore.rescore(pa, ea, self.root / "a")
        b_source = self.root / "b-source"
        b_source.mkdir()
        pb = b_source / "predictions.jsonl"
        pb.write_text(json.dumps(rows_b[0]) + "\n")
        eb = b_source / "evaluation.json"
        eb.write_text(json.dumps({"status": "complete", "model": "medgemma", "split": "val",
                                  "predictions_sha256": hashlib.sha256(pb.read_bytes()).hexdigest(),
                                  "counts": {"images": 1, "fracture_boxes": 2},
                                  "protocol": {"selected_on": "val", "identity": {
                                      "model": "medgemma", "raw_manifest_sha256": rescore.MANIFEST_SHA256}}}))
        result_b = rescore.rescore(pb, eb, self.root / "b")
        self.assertEqual(result_a["counts"]["TP"], 2)
        self.assertEqual(result_a["metrics"], result_b["metrics"])

    def test_threshold_and_float32_epsilon_are_numeric_and_applied(self):
        self.assertEqual(rescore.IOU_THRESHOLD, 0.50)
        self.assertEqual(rescore.float32(rescore.EPSILON), 1.0000000116860974e-07)
        at_half = rescore.iou_float32([0, 0, 0.2, 0.1], [0, 0, 0.1, 0.1])
        above_half = rescore.iou_float32([0, 0, 0.2, 0.1], [0, 0, 0.101, 0.1])
        self.assertLess(at_half, rescore.IOU_THRESHOLD)
        self.assertGreater(above_half, rescore.IOU_THRESHOLD)
        self.assertEqual(rescore.matched_count([[0, 0, 0.1, 0.1]], [[0, 0, 0.2, 0.1]]), 0)
        self.assertEqual(rescore.matched_count([[0, 0, 0.101, 0.1]], [[0, 0, 0.2, 0.1]]), 1)

    def test_rejects_source_count_and_ground_truth_geometry_failures(self):
        predictions, evaluation = self.write_inputs([row("one")])
        report = json.loads(evaluation.read_text())
        del report["counts"]
        evaluation.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, "counts.images"):
            rescore.rescore(predictions, evaluation, self.root / "missing-counts")
        predictions, evaluation = self.write_inputs([row("one")])
        report = json.loads(evaluation.read_text())
        report["counts"]["images"] = 0
        evaluation.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, "counts.images"):
            rescore.rescore(predictions, evaluation, self.root / "zero-images")
        predictions, evaluation = self.write_inputs([row("one")])
        report = json.loads(evaluation.read_text())
        report["counts"]["fracture_boxes"] = -1
        evaluation.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, "counts.fracture_boxes"):
            rescore.rescore(predictions, evaluation, self.root / "negative-fractures")
        predictions, evaluation = self.write_inputs([row("one")])
        report = json.loads(evaluation.read_text())
        report["counts"]["images"] = 2
        evaluation.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, "counts do not match"):
            rescore.rescore(predictions, evaluation, self.root / "wrong-counts")
        for bad_gt in ([10, 0, 10, 20], [20, 0, 10, 20], [-1, 0, 10, 20], [0, 0, 1001, 20]):
            with self.subTest(bad_gt=bad_gt):
                predictions, evaluation = self.write_inputs([row("one", gt=[bad_gt])])
                with self.assertRaisesRegex(ValueError, "gt_xyxy"):
                    rescore.rescore(predictions, evaluation, self.root / ("bad-gt-%s" % bad_gt[0]))

    def test_rejects_broken_output_symlink_and_predictions_parent(self):
        predictions, evaluation = self.write_inputs([row("one")])
        broken = self.root / "broken-output"
        try:
            broken.symlink_to(self.root / "missing-target")
        except (NotImplementedError, OSError) as exc:
            self.skipTest("symlinks unavailable: %s" % exc)
        with self.assertRaisesRegex(ValueError, "already exists"):
            rescore.rescore(predictions, evaluation, broken)
        other = self.root / "other-evaluation"
        other.mkdir()
        external_evaluation = other / "evaluation.json"
        external_evaluation.write_text(evaluation.read_text())
        with self.assertRaisesRegex(ValueError, "predictions folder"):
            rescore.rescore(predictions, external_evaluation, predictions.parent / "nested")
        with self.assertRaisesRegex(ValueError, "source evaluation folder"):
            rescore.rescore(predictions, external_evaluation, other / "nested")

    def test_cli_writes_only_four_point_estimates(self):
        predictions, evaluation = self.write_inputs([row("one")])
        destination = self.root / "cli-output"
        completed = subprocess.run([sys.executable, str(ROOT / "rescore_medgemma.py"), "--predictions", str(predictions),
                                    "--source-evaluation", str(evaluation), "--output", str(destination)],
                                   text=True, capture_output=True, check=False)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual((destination / "metrics.csv").read_text().splitlines()[1:],
                         ["precision,0.0", "lesion_recall,0.0", "F1,0.0", "false_positives_per_image,0.0"])
        payload = json.loads((destination / "rescore.json").read_text())
        self.assertTrue(payload["not_a_protocol_validation_report"])
        self.assertIsNone(payload["confidence_intervals"])
        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(payload["method_version"], rescore.METHOD_VERSION)
        self.assertEqual(payload["rescore_source_sha256"], rescore.sha256(ROOT / "rescore_medgemma.py"))
        self.assertEqual(payload["python"]["executable"], sys.executable)
        self.assertEqual(payload["python"]["version"], sys.version)

    @unittest.skipUnless(os.environ.get("FRACTURE_MEDGEMMA_RESULTS"),
                         "set FRACTURE_MEDGEMMA_RESULTS to a real MedGemma evaluation directory")
    def test_optional_real_fixture_regression(self):
        source = Path(os.environ["FRACTURE_MEDGEMMA_RESULTS"])
        predictions = source / "predictions.jsonl"
        evaluation = source / "evaluation.json"
        destination = self.root / "real-output"
        result = rescore.rescore(predictions, evaluation, destination)
        self.assertEqual(result["counts"]["images"], 2895)
        self.assertEqual(result["counts"]["ground_truth_boxes"], 2734)
        self.assertEqual(result["counts"]["generated_boxes"], 2050)
        self.assertEqual(result["counts"]["TP"], 1369)
        self.assertEqual(result["counts"]["FP"], 681)
        self.assertEqual(result["counts"]["FN"], 1365)
        expected = {"precision": 1369 / 2050, "lesion_recall": 1369 / 2734,
                    "F1": 2738 / 4784, "false_positives_per_image": 681 / 2895}
        self.assertEqual(result["metrics"], expected)


if __name__ == "__main__":
    unittest.main()
