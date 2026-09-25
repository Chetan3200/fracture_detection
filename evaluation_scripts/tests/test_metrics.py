"""CPU-only metric regressions against exact current attachment and Ultralytics 8.4.152.
Run: python -m unittest discover -s tests -p test_metrics.py -v
No model, image, CUDA, network, or Ultralytics installation is needed.
Optional offline JSONL comparisons run only if the original saved files are present.
"""
import ast
from contextlib import redirect_stdout
from copy import deepcopy
import hashlib
import importlib.util
import inspect
import io
import json
from pathlib import Path
import tempfile
import textwrap
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import warnings

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

common = load_module("evaluation_common_metrics_test", HERE.parent / "evaluation_common.py")
ref = load_module("reference_ultralytics_metrics_test", HERE / "reference_ultralytics.py")
METRICS = ["AP50_95", "AP50", "lesion_recall", "false_positives_per_image", "precision", "F1"]
SAVED = Path("/Users/chetan/.aside/u/0/sessions/2026-09-17_yPN42v0eEXQ7a1u8/tmp/evaluation_review")


def reference_ap_matches(ious):
    matcher = SimpleNamespace(iouv=torch.linspace(0.5, 0.95, 10))
    return ref.match_predictions(matcher, torch.zeros(ious.shape[1], dtype=torch.int64),
                                 torch.zeros(ious.shape[0], dtype=torch.int64),
                                 torch.from_numpy(ious)).numpy()


def reference_record(stem, patient, gt, pred, scores):
    """Original record-building logic after the explicitly emulated collector bounds."""
    gt = np.asarray(gt, dtype=np.float32).reshape(-1, 4)
    pred = np.asarray(pred, dtype=np.float32).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float32)
    order = np.argsort(-scores, kind="stable")
    # Original detector enforces these bounds before record construction.
    order = order[scores[order] > ref.COLLECTION["conf"]][:ref.COLLECTION["max_det"]]
    pred, scores = pred[order], scores[order]
    ious = ref.box_iou(torch.from_numpy(gt), torch.from_numpy(pred)).numpy()
    return dict(stem=stem, patient=int(patient), n_gt=len(gt), scores=scores,
                ap_correct=reference_ap_matches(ious), op_correct=ref.operating_matches(ious))


def fixture():
    # Unequal image counts, deliberately unsorted patient insertion order, and two negative patients.
    inputs = [
        ("30_a", 30, [[0, 0, 10, 10]], [[0, 0, 10, 10]], [.8]),
        ("10_a", 10, [], [[0, 0, 10, 10]], [.7]),
        ("30_b", 30, [[0, 0, 10, 10]], [], []),
        ("20_a", 20, [], [], []),
        ("10_b", 10, [], [[1, 1, 2, 2]], [.7]),
    ]
    return ([common.make_record(*x) for x in inputs], [reference_record(*x) for x in inputs])


def cohort_for(records):
    return pd.DataFrame([dict(filestem=r["stem"], patient_id=r["patient"],
                              fracture_count=r["n_gt"],
                              sample_type="positive" if r["n_gt"] else "negative",
                              projection="AP") for r in records])


class MetricTests(unittest.TestCase):
    def setUp(self):
        # Any accidental GPU benchmarking is a test failure, never a silent mocked benchmark.
        for name in ("synchronize", "empty_cache", "reset_peak_memory_stats", "get_device_name"):
            blocker = patch.object(torch.cuda, name, side_effect=AssertionError("GPU work forbidden in metric tests"))
            blocker.start()
            self.addCleanup(blocker.stop)

    def assert_metrics_equal(self, actual, expected):
        self.assertEqual(list(actual), METRICS)
        self.assertEqual(list(expected), METRICS)
        np.testing.assert_allclose([actual[k] for k in METRICS], [expected[k] for k in METRICS],
                                   rtol=0, atol=1e-14, equal_nan=True)

    def assert_records_equal(self, actual, expected):
        for a, b in zip(actual, expected, strict=True):
            self.assertEqual((a["stem"], a["patient"], a["n_gt"]), (b["stem"], b["patient"], b["n_gt"]))
            for key in ("scores", "op_correct", "ap_correct"):
                np.testing.assert_array_equal(a[key], b[key])

    def test_exact_current_attachment_and_vendored_function_provenance(self):
        current = ref.PROVENANCE["current_attachment"]
        self.assertEqual(current["path"], "/Users/chetan/.aside/u/0/sessions/2026-09-18_YgeuHthvdMJU1XN4/attachments/grazpedwri_evaluate_yolo26.py")
        self.assertEqual(current["sha256"], "e690b96d66d88f669aee017296c1a04806d540487a7b27847170d5cf08d8ef85")
        wanted = {"box_iou", "smooth", "compute_ap", "ap_per_class", "match_predictions",
                  "operating_matches", "choose_threshold", "summarize"}
        found = set()
        for source in ref.PROVENANCE.values():
            path = Path(source["path"])
            upstream = None
            if path.exists():
                data = path.read_bytes()
                self.assertEqual(hashlib.sha256(data).hexdigest(), source["sha256"])
                upstream = data.decode().splitlines(keepends=True)
            for name, item in source["functions"].items():
                found.add(name)
                extracted = textwrap.dedent(inspect.getsource(getattr(ref, name))).rstrip() + "\n"
                self.assertEqual(hashlib.sha256(extracted.encode()).hexdigest(), item["dedented_source_sha256"])
                if upstream is not None:
                    original = textwrap.dedent("".join(upstream[item["lineno"]-1:item["end_lineno"]])).rstrip() + "\n"
                    self.assertEqual(extracted, original)
        self.assertEqual(found, wanted)
        self.assertTrue(ref.checks.check_version(np.__version__, ">=2.0") == (int(np.__version__.split(".")[0]) >= 2))
        self.assertEqual(ref.COLLECTION["quantize"], 16)  # Valid 8.4.152 option; not renamed to half.

    def test_pairwise_iou_random_boxes_and_empty_shapes_exact(self):
        rng = np.random.default_rng(771)
        for n, m in [(0, 0), (0, 9), (7, 0), (1, 1), (6, 9), (31, 54)]:
            for _ in range(8):
                a = rng.uniform(-100, 100, (n, 4)).astype(np.float32)
                b = rng.uniform(-100, 100, (m, 4)).astype(np.float32)
                a[:, 2:] = a[:, :2] + np.abs(a[:, 2:]) + .01
                b[:, 2:] = b[:, :2] + np.abs(b[:, 2:]) + .01
                expected = ref.box_iou(torch.from_numpy(a), torch.from_numpy(b)).numpy()
                np.testing.assert_array_equal(common.pairwise_iou(a, b), expected)

    def test_exact_torch_iou_boundaries_and_neighbors(self):
        thresholds = torch.linspace(.5, .95, 10).numpy()
        # One GT per prediction prevents duplicate suppression obscuring threshold boundaries.
        for values in (thresholds, np.nextafter(thresholds, np.float32(-np.inf)),
                       np.nextafter(thresholds, np.float32(np.inf))):
            ious = np.diag(values).astype(np.float32)
            np.testing.assert_array_equal(common.ap_matches(ious), reference_ap_matches(ious))
            for index, value in enumerate(values):
                np.testing.assert_array_equal(common.ap_matches(ious)[index], value >= thresholds)
        operating = np.diag(np.array([np.nextafter(np.float32(.5), np.float32(0)), .5,
                                     np.nextafter(np.float32(.5), np.float32(1))], dtype=np.float32))
        np.testing.assert_array_equal(common.operating_matches(operating), [False, True, True])

    def test_matching_random_ties_and_ap_operating_difference(self):
        rng = np.random.default_rng(871)
        for _ in range(150):
            ious = rng.choice([0, .49, .5, .55, .7, .9, 1], (int(rng.integers(0, 8)), int(rng.integers(0, 20)))).astype(np.float32)
            np.testing.assert_array_equal(common.ap_matches(ious), reference_ap_matches(ious))
            np.testing.assert_array_equal(common.operating_matches(ious), ref.operating_matches(ious))
        ious = np.array([[.6, .9]], dtype=np.float32)
        np.testing.assert_array_equal(common.operating_matches(ious), [True, False])
        # Keep the pinned reference's duplicate-removal order; it is not simply
        # "highest IoU wins" after the successive np.unique operations.
        np.testing.assert_array_equal(common.ap_matches(ious), reference_ap_matches(ious))
        # No ties: operating matching can fall back to the second GT, whereas AP
        # deduplicates each prediction's chosen GT and drops that second match.
        ious = np.array([[.9, .8], [0, .7]], dtype=np.float32)
        np.testing.assert_array_equal(common.operating_matches(ious), [True, True])
        np.testing.assert_array_equal(common.ap_matches(ious)[:, 0], [True, False])
        np.testing.assert_array_equal(common.ap_matches(ious), reference_ap_matches(ious))
        self.assertFalse(np.array_equal(common.ap_matches(ious)[:, 0], common.operating_matches(ious)))

    def test_128_random_cohorts_six_metrics_thresholds_and_score_ties(self):
        rng = np.random.default_rng(310519)
        ties = np.array([0, .001, .002, .01, .2, .5, .8, 1], dtype=np.float32)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for trial in range(128):
                actual, expected = [], []
                for image in range(int(rng.integers(2, 10))):
                    n_gt = int(rng.integers(1 if image == 0 else 0, 6))
                    n_pred = int(rng.integers(0, 18))
                    gt = rng.uniform(0, 100, (n_gt, 4)).astype(np.float32)
                    gt[:, 2:] = gt[:, :2] + rng.uniform(2, 40, (n_gt, 2))
                    pred = rng.uniform(0, 100, (n_pred, 4)).astype(np.float32)
                    pred[:, 2:] = pred[:, :2] + rng.uniform(2, 40, (n_pred, 2))
                    for j in range(min(n_gt, n_pred)):
                        pred[j] = gt[j] + rng.uniform(-2, 2, 4).astype(np.float32)
                        pred[j, 2:] = np.maximum(pred[j, 2:], pred[j, :2] + .01)
                    if n_pred > 2 and n_gt:
                        pred[-1] = gt[0]  # repeated lesion boxes exercise duplicate FP handling.
                    scores = rng.choice(ties, n_pred) if trial % 2 else rng.uniform(.00101, 1, n_pred).astype(np.float32)
                    inputs = (f"{image:04}_{trial}", image // 2, gt, pred, scores)
                    actual.append(common.make_record(*inputs))
                    expected.append(reference_record(*inputs))
                with self.subTest(cohort=trial):
                    self.assert_records_equal(actual, expected)
                    cutoff, table = common.choose_threshold(actual)
                    original_cutoff, original_table = ref.choose_threshold(expected)
                    self.assertEqual(cutoff, original_cutoff)
                    pd.testing.assert_frame_equal(table, original_table, check_exact=True)
                    for threshold in (cutoff, .001, .5, 1):
                        self.assert_metrics_equal(common.summarize(actual, threshold), ref.summarize(expected, threshold))

    def test_threshold_f1_tie_prefers_highest_cutoff_and_strict_greater(self):
        records = [dict(stem="a", patient=1, n_gt=2, scores=np.array([.9, .8, .7, .6], dtype=np.float32),
                        op_correct=np.array([1, 0, 0, 1], bool), ap_correct=np.zeros((4, 10), bool))]
        cutoff, sweep = common.choose_threshold(records)
        self.assertEqual(cutoff, float(np.float32(.8)))
        self.assertEqual(sweep.F1.iloc[1], sweep.F1.iloc[-1])
        self.assertAlmostEqual(common.summarize(records, cutoff)["F1"], 2/3)
        self.assertEqual(common.summarize(records, cutoff)["precision"], 1)
        self.assert_metrics_equal(common.summarize(records, cutoff), ref.summarize(records, cutoff))
        # Equal scores across distinct patients are included/excluded together.
        tied = [dict(stem=str(i), patient=i, n_gt=1, scores=np.array([.5], np.float32),
                     op_correct=np.array([i == 0]), ap_correct=np.zeros((1, 10), bool)) for i in range(2)]
        chosen, table = common.choose_threshold(tied)
        self.assertEqual(chosen, .001)
        self.assertEqual(len(table), 2)
        self.assertEqual(common.summarize(tied, .5)["lesion_recall"], 0)
        self.assertEqual(common.summarize(tied, chosen)["precision"], .5)

    def test_empty_predictions_no_gt_and_all_incorrect(self):
        positive = common.make_record("p", 1, [[0, 0, 10, 10]], [], [])
        with self.assertWarns(UserWarning):
            cutoff, table = common.choose_threshold([positive])
        self.assertEqual(cutoff, 1)
        self.assertTrue(table.empty)
        self.assertEqual(common.summarize([positive], cutoff), dict.fromkeys(METRICS, 0.0))
        negative = common.make_record("n", 2, [], [], [])
        result = common.summarize([negative], .5)
        for name in ["AP50", "AP50_95", "lesion_recall", "F1"]:
            self.assertTrue(np.isnan(result[name]))
        self.assertEqual(result["precision"], 0)
        self.assertEqual(result["false_positives_per_image"], 0)
        self.assert_metrics_equal(result, ref.summarize([negative], .5))
        with self.assertRaises(ValueError):
            common.choose_threshold([negative])
        with self.assertRaises(ValueError):
            common.summarize([], .5)
        miss = common.make_record("m", 3, [[0, 0, 10, 10]], [[20, 20, 30, 30]], [.9])
        with self.assertWarns(UserWarning):
            self.assertEqual(common.choose_threshold([miss])[0], 1)
        self.assertEqual(common.summarize([miss], .001)["AP50"], 0)

    def test_duplicate_fp_negative_images_collection_floor_and_cap(self):
        p = common.make_record("p", 1, [[0, 0, 10, 10]], [[0, 0, 10, 10]]*2, [.8, .7])
        n = common.make_record("n", 2, [], [[0, 0, 10, 10]], [.6])
        result = common.summarize([p, n], .001)
        self.assertEqual(result["lesion_recall"], 1)
        self.assertEqual(result["precision"], 1/3)
        self.assertEqual(result["false_positives_per_image"], 1)
        self.assertEqual(result["F1"], .5)
        floor = np.float32(.001)
        r = common.make_record("floor", 1, [], [[0, 0, 1, 1]]*4,
                               [0, np.nextafter(floor, np.float32(0)), floor, np.nextafter(floor, np.float32(1))])
        self.assertEqual(len(r["scores"]), 1)
        for count in [299, 300, 301, 350]:
            boxes = np.array([[i, 0, i+1, 1] for i in range(count)], np.float32)
            capped = common.make_record("c", 1, [], boxes, [.5]*count)
            self.assertEqual(len(capped["scores"]), min(count, 300))
            self.assertEqual(capped["at_max_det"], count >= 300)
            np.testing.assert_array_equal(capped["pred_xyxy"], boxes[:300])

    def test_bootstrap_twenty_fixed_patient_draws_matches_reference_arrays(self):
        actual, expected = fixture()
        patients = {}
        for record in expected:
            patients.setdefault(record["patient"], []).append(record)
        ids = np.array(sorted(patients))
        rng = np.random.default_rng(2026)
        draws, sizes = [], []
        # Literal current attachment algorithm, lines 796-823; model and cutoff stay fixed.
        for _ in range(20):
            sampled = rng.choice(ids, size=len(ids), replace=True)
            resample = [record for pid in sampled for record in patients[pid]]
            sizes.append(len(resample))
            draws.append(ref.summarize(resample, .5))
        reference = pd.DataFrame(draws)
        with patch.object(common, "choose_threshold", side_effect=AssertionError("Do not retune bootstrap")):
            result = common.bootstrap(actual, .5, 20, 2026)
        pd.testing.assert_frame_equal(result, reference, check_exact=True)
        self.assertGreater(len(set(sizes)), 1)  # patients, not fixed-count images, were resampled.
        self.assertTrue(result.AP50.isna().any())
        self.assertTrue(result.AP50.notna().any())
        pd.testing.assert_frame_equal(result.quantile([.025, .975]), reference.quantile([.025, .975]), check_exact=True)
        pd.testing.assert_series_equal(result.notna().sum(), reference.notna().sum())

    def test_subgroups_missing_flags_remain_unknown_and_counts_are_correct(self):
        records = [common.make_record(str(i), i//2, [[0, 0, 10, 10]] if i % 2 == 0 else [], [], []) for i in range(7)]
        cohort = cohort_for(records)
        cohort["cast"] = [1, 0, None, np.nan, "bad", 2, "1"]
        cohort["diagnosis_uncertain"] = [None, 0, 1, "", 2, "0", "bad"]
        cohort["projection"] = [" AP ", "", None, "LAT", "custom", "AP", "LAT"]
        cohort["age"] = [0, 6, 11, 16, None, "bad", -1]
        cohort["gender"] = ["F", None, "M", "F", "M", "F", "M"]
        result = common.subgroups(cohort, records, .5)
        for family, unknown_count in [("cast", 4), ("diagnosis_uncertain", 4)]:
            rows = result[result.subgroup_type.eq(family)].set_index("subgroup")
            self.assertEqual(rows.loc["unknown", "images"], unknown_count)
            self.assertEqual(rows.images.sum(), 7)
            self.assertEqual(rows.fracture_boxes.sum(), 4)
        views = result[result.subgroup_type.eq("view")].set_index("subgroup")
        self.assertEqual(views.loc["unknown", "images"], 2)
        self.assertIn("custom", views.index)
        ages = result[result.subgroup_type.eq("age_years")].set_index("subgroup")
        self.assertEqual(ages.loc["unknown", "images"], 3)
        for label in ["0-<6", "6-<11", "11-<16", "16+"]:
            self.assertEqual(ages.loc[label, "images"], 1)

    def test_protocol_identity_and_mismatch_guards(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)/"source.py"
            source.write_text("# source A\n")
            checkpoint = {"weights_sha256": "a"*64, "config_sha256": "b"*64, "local_path": "/first"}
            settings = {"imgsz": 640, "quantize": 16}
            identity = common.protocol_identity("yolo26", checkpoint, settings, [source])
            moved = {**checkpoint, "local_path": "/other", "remote_repo": "example/weights"}
            self.assertEqual(identity, common.protocol_identity("yolo26", moved, settings, [source]))
            protocol = {"selected_on": "val", "confidence_cutoff": .5, "identity": identity}
            report = {"status": "complete", "split": "val", "protocol": protocol}
            saved = Path(tmp)/"evaluation.json"
            saved.write_text(json.dumps(report))
            args = SimpleNamespace(split="test", protocol=saved)
            self.assertEqual(common.frozen_protocol(args, identity), protocol)
            self.assertIsNone(common.frozen_protocol(SimpleNamespace(split="val"), identity))
            variants = [common.protocol_identity("different-model", checkpoint, settings, [source]),
                        common.protocol_identity("yolo26", {**checkpoint, "weights_sha256": "c"*64}, settings, [source]),
                        common.protocol_identity("yolo26", checkpoint, {**settings, "imgsz": 960}, [source])]
            source.write_text("# source B\n")
            variants.append(common.protocol_identity("yolo26", checkpoint, settings, [source]))
            for changed in variants:
                with self.assertRaisesRegex(ValueError, "changed"):
                    common.frozen_protocol(args, changed)
            for field, value in [("status", "partial"), ("split", "test")]:
                invalid = deepcopy(report); invalid[field] = value
                saved.write_text(json.dumps(invalid))
                with self.assertRaises(ValueError):
                    common.frozen_protocol(args, identity)
            for threshold in [-1, 0, 1.1, "0.5", float("nan")]:
                invalid = deepcopy(report); invalid["protocol"]["confidence_cutoff"] = threshold
                saved.write_text(json.dumps(invalid))
                with self.assertRaises(ValueError):
                    common.frozen_protocol(args, identity)
            invalid = deepcopy(report); invalid["protocol"]["selected_on"] = "test"
            saved.write_text(json.dumps(invalid))
            with self.assertRaises(ValueError):
                common.frozen_protocol(args, identity)

    def test_full_cpu_report_only_four_default_files_and_frozen_test_cutoff(self):
        records, original = fixture()
        # An actual capped negative-image record checks report cap counts, not just make_record.
        capped_input = ("10_a", 10, [], [[0, 0, 10, 10]]*301, [.7]*301)
        records[1], original[1] = common.make_record(*capped_input), reference_record(*capped_input)
        cohort = cohort_for(records)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = SimpleNamespace(split="val", model_name="synthetic", bootstraps=100, bootstrap_seed=2026,
                                   diagnostics=False, protocol=None)
            identity = {"synthetic": True}
            bench = {"synthetic": True, "timed_runs": 100, "note": "fixture, no hardware benchmark executed"}
            val = root/"val"; val.mkdir()
            (val/"predictions.jsonl").write_text("{\"fixture\":true}\n")
            with warnings.catch_warnings(), redirect_stdout(io.StringIO()):
                warnings.simplefilter("ignore")
                common.finish(args, val, cohort, records, {}, identity, None, bench)
            self.assertEqual({p.name for p in val.iterdir()}, {"predictions.jsonl", "metrics.csv", "subgroups.csv", "evaluation.json"})
            report = json.loads((val/"evaluation.json").read_text())
            self.assertEqual(report["status"], "complete")
            self.assertEqual(report["counts"]["images"], 5)
            self.assertEqual(report["counts"]["patients"], 3)
            self.assertEqual(report["counts"]["fracture_boxes"], 2)
            self.assertEqual(report["counts"]["negative_images"], 3)
            self.assertEqual(report["counts"]["images_at_max_det"], 1)
            cutoff = report["protocol"]["confidence_cutoff"]
            metrics = pd.read_csv(val/"metrics.csv").set_index("metric")
            expected = ref.summarize(original, cutoff)
            np.testing.assert_allclose(metrics.loc[METRICS, "estimate"], [expected[k] for k in METRICS], rtol=0, atol=1e-14)
            self.assertLess(metrics.loc["AP50", "valid_bootstrap_replicates"], 100)
            self.assertEqual(metrics.loc["precision", "valid_bootstrap_replicates"], 100)
            self.assertEqual(report["benchmark"], bench)
            test = root/"test"; test.mkdir()
            (test/"predictions.jsonl").write_text("{\"fixture\":true}\n")
            args.split, args.protocol = "test", val/"evaluation.json"
            frozen = common.frozen_protocol(args, identity)
            with patch.object(common, "choose_threshold", side_effect=AssertionError("Test tuning forbidden")), warnings.catch_warnings(), redirect_stdout(io.StringIO()):
                warnings.simplefilter("ignore")
                common.finish(args, test, cohort, records, {}, identity, frozen, bench)
            result = json.loads((test/"evaluation.json").read_text())
            self.assertEqual(result["protocol"], report["protocol"])
            self.assertEqual(len(list(test.iterdir())), 4)
            self.assertEqual(result["validation_protocol_sha256"], common.sha256(args.protocol))

    def test_optional_saved_640_and_960_prediction_datasets(self):
        paths = [SAVED/f"{size}_predictions.jsonl" for size in (640, 960)]
        if not all(p.is_file() for p in paths):
            self.skipTest("Original offline prediction datasets are not bundled with these tests")
        for path in paths:
            actual, expected = [], []
            for line in path.read_text().splitlines():
                item = json.loads(line)
                inputs = (item["filestem"], item["patient_id"], item["gt_xyxy"], item["pred_xyxy"], item["scores"])
                actual.append(common.make_record(*inputs))
                expected.append(reference_record(*inputs))
            with self.subTest(dataset=path.name):
                self.assert_records_equal(actual, expected)
                cutoff, sweep = common.choose_threshold(actual)
                original_cutoff, original_sweep = ref.choose_threshold(expected)
                self.assertEqual(cutoff, original_cutoff)
                pd.testing.assert_frame_equal(sweep, original_sweep, check_exact=True)
                point = common.summarize(actual, cutoff)
                self.assert_metrics_equal(point, ref.summarize(expected, original_cutoff))
                print("OFFLINE_REFERENCE", json.dumps({"dataset": path.name, "images": len(actual),
                    "patients": len({r["patient"] for r in actual}), "cutoff": cutoff, **point}, sort_keys=True))


if __name__ == "__main__":
    unittest.main(verbosity=2)
