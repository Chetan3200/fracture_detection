"""CPU-only tests of production data-validation, parsing and loss-masking logic.
No model download, GPU, torch import, or real dataset is required.
Run beside train_medgemma.py: python test_train_medgemma.py
This is not an end-to-end GPU training test; use --smoke for that.
"""
from pathlib import Path
from unittest.mock import patch
import copy
import csv
import hashlib
import importlib.util
import io
import json
import random
import struct
import tempfile
import unittest
import zlib

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SCRIPT = ROOT / "train_medgemma.py"
spec = importlib.util.spec_from_file_location("training_script_under_test", SCRIPT)
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)
COUNTS = {
    "train": {"positive": 2, "negative": 1, "excluded": 1},
    "val": {"positive": 1, "negative": 1, "excluded": 1},
    "test": {"positive": 1, "negative": 1, "excluded": 0},
}
# Counter equality before Python 3.10 distinguishes missing keys from zero keys.
# Omit empty SYNTHETIC categories; every real frozen category remains nonzero.
PATCH_COUNTS = {split: {kind: n for kind, n in counts.items() if n}
                for split, counts in COUNTS.items()}


def tiny_rgb_png():
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 2, 1, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(b"\x00\x00\x00\x00\xff\xff\xff")) + chunk(b"IEND", b""))


def answer(boxes):
    return "Final Answer: ```json\n" + json.dumps(boxes, separators=(",", ":")) + "\n```"


def fixture(root):
    root = Path(root)
    prompt = "Locate all visible fractures. Use yxyx coordinates in [0,1000]."
    rows = []
    for split, counts in COUNTS.items():
        for category, n in counts.items():
            for index in range(n):
                pid = str(len(rows) + 1)
                stem = f"{int(pid):04d}_study_{index}"
                boxes = [[0, .5, .6, .2, .1]] if category == "positive" else []
                rows.append({"filestem": stem, "patient_id": pid, "split": split,
                             "sample_type": category, "fracture_labels_json": json.dumps(boxes)})
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    raw = buffer.getvalue().encode()
    h = hashlib.sha256(raw).hexdigest()
    (root / "split_manifest.csv").write_bytes(raw)
    (root / "prompt.txt").write_text(prompt)
    (root / "dataset_info.json").write_text(json.dumps({"manifest_sha256": h, "prompt_sha256": app.digest(prompt.encode())}))
    (root / "processor_requirements.json").write_text(json.dumps({"model_id": app.MODEL_ID, "do_pan_and_scan": False}))
    for split in ("train", "val"):
        records, inputs = [], []
        for row in rows:
            if row["split"] != split or row["sample_type"] == "excluded":
                continue
            image = f"images/{split}/{row['filestem']}.png"
            (root / image).parent.mkdir(parents=True, exist_ok=True)
            (root / image).write_bytes(tiny_rgb_png())
            base = {"image_id": row["filestem"], "patient_id": row["patient_id"], "split": split,
                    "image": image, "messages": [app.expected_user_message(prompt)]}
            inputs.append(copy.deepcopy(base))
            base["messages"].append({"role": "assistant", "content": [{"type": "text", "text": answer(app.manifest_targets(row))}]})
            records.append(base)
        (root / f"{split}.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n")
        if split == "val":
            (root / "val_inputs.jsonl").write_text("\n".join(json.dumps(r) for r in inputs) + "\n")
    # Deliberately NO test image, test JSONL or test reference is created.
    return h


class Tests(unittest.TestCase):
    def test_masks_only_assistant_and_keeps_stop_token(self):
        prompt = [2, 106, 99, 262144, 262144, 107, 106, 55, 108]
        ids = prompt + [80, 81, 107, 0, 0]
        mask = [1] * (len(prompt) + 3) + [0, 0]
        expected = [-100] * len(prompt) + [80, 81, 107, -100, -100]
        self.assertEqual(app.masked_labels(ids, prompt, mask), expected)
        self.assertEqual(ids[-2:], [0, 0])  # no in-place input mutation

    def test_masking_fails_closed(self):
        for ids, prompt, mask in [([1,2,3], [1,9], [1,1,1]), ([1,2], [1,2], [1,1]),
                                  ([1,2,3], [1,2], [0,1,1]), ([1,2,3], [1,2], [1,1])]:
            with self.assertRaises(ValueError):
                app.masked_labels(ids, prompt, mask)

    def test_strict_parser(self):
        boxes = [{"box_2d": [550,400,650,600], "label": "fracture"}]
        self.assertEqual(app.parse_answer(answer(boxes)), boxes)
        self.assertEqual(app.parse_answer(answer([])), [])
        invalid = ["[]", "reasoning\n" + answer([]), answer([]) + "commentary",
                   answer([{"box_2d":[0,0,0,5],"label":"fracture"}]),
                   answer([{"box_2d":[0,0,1001,5],"label":"fracture"}]),
                   answer([{"box_2d":[False,0,5,5],"label":"fracture"}]),
                   answer([{"box_2d":[0,0,5,5],"label":"normal"}]),
                   answer([{"box_2d":[float('nan'),0,5,5],"label":"fracture"}])]
        for text in invalid:
            with self.assertRaises((ValueError, TypeError)):
                app.parse_answer(text)

    def test_coordinate_conversion(self):
        result = app.manifest_targets({"fracture_labels_json": "[[0,0.5,0.6,0.2,0.1]]"})
        self.assertEqual(result, [{"box_2d":[550.,400.,650.,600.], "label":"fracture"}])
        rng = random.Random(42)
        for _ in range(1000):
            w,h = rng.uniform(.01,.9),rng.uniform(.01,.9)
            x,y = rng.uniform(w/2,1-w/2),rng.uniform(h/2,1-h/2)
            actual = app.manifest_targets({"fracture_labels_json":json.dumps([[0,x,y,w,h]])})[0]["box_2d"]
            expected = [y-h/2,x-w/2,y+h/2,x+w/2]
            self.assertTrue(all(abs(a/1000-b) <= 5.00001e-6 for a,b in zip(actual,expected)))

    def test_stable_balanced_subset(self):
        records = [{"image_id": str(i)} for i in range(100)]
        kinds = {str(i): "positive" if i % 2 else "negative" for i in range(100)}
        first = app.balanced_subset(records, 16, kinds, "fixed")
        second = app.balanced_subset(list(reversed(records)), 16, kinds, "fixed")
        self.assertEqual(first, second)
        self.assertEqual(len({r["image_id"] for r in first}),16)
        self.assertEqual(sum(kinds[r["image_id"]] == "positive" for r in first),8)
        self.assertEqual(app.balanced_subset(records,0,kinds,"fixed"),[])

    def test_load_bundle_without_any_test_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            h = fixture(tmp)
            with patch.object(app,"EXPECTED_COUNTS",PATCH_COUNTS):
                bundle = app.load_bundle(tmp, expected_manifest=h)
            self.assertEqual(len(bundle["datasets"]["train"]),3)
            self.assertEqual(len(bundle["datasets"]["val"]),2)
            self.assertEqual(len(bundle["val_inputs"]),2)
            self.assertTrue(all(r["messages"][0] == app.expected_user_message(bundle["prompt"]) for r in bundle["val_inputs"]))

    def test_answer_leakage_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            h = fixture(tmp)
            p = Path(tmp) / "val_inputs.jsonl"
            records = app.read_jsonl(p)
            records[0]["messages"].append({"role":"assistant","content":[{"type":"text","text":answer([])}]})
            p.write_text("\n".join(json.dumps(r) for r in records))
            with patch.object(app,"EXPECTED_COUNTS",PATCH_COUNTS), self.assertRaisesRegex(ValueError,"leaked"):
                app.load_bundle(tmp, expected_manifest=h)

    def test_changed_target_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            h = fixture(tmp)
            p = Path(tmp) / "train.jsonl"
            records = app.read_jsonl(p)
            records[0]["messages"][1]["content"][0]["text"] = answer([])
            p.write_text("\n".join(json.dumps(r) for r in records))
            with patch.object(app,"EXPECTED_COUNTS",PATCH_COUNTS), self.assertRaisesRegex(ValueError,"differs"):
                app.load_bundle(tmp, expected_manifest=h)

    def test_changed_prompt_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            h = fixture(tmp)
            (Path(tmp)/"prompt.txt").write_text("changed")
            with self.assertRaisesRegex(ValueError,"Prompt differs"):
                app.load_bundle(tmp, expected_manifest=h)

    def test_resume_requires_scheduler_and_every_rank_rng(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "checkpoint-4"
            checkpoint.mkdir()
            files = ["training_args.bin", "adapter_config.json", "adapter_model.safetensors", "optimizer.pt",
                     "scheduler.pt", "rng_state_0.pth", "rng_state_1.pth", "rng_state_2.pth", "rng_state_3.pth"]
            for name in files:
                (checkpoint / name).write_bytes(b"nonempty fixture")
            (checkpoint / "trainer_state.json").write_text('{"global_step":4}')
            self.assertEqual(app.validate_checkpoint(checkpoint,root,4),checkpoint.resolve())
            (checkpoint / "scheduler.pt").unlink()
            with self.assertRaisesRegex(ValueError,"scheduler.pt"):
                app.validate_checkpoint(checkpoint,root,4)
            (checkpoint / "scheduler.pt").write_bytes(b"nonempty fixture")
            (checkpoint / "rng_state_3.pth").unlink()
            with self.assertRaisesRegex(ValueError,"rng_state_3.pth"):
                app.validate_checkpoint(checkpoint,root,4)

    def test_wrong_manifest_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture(tmp)
            with self.assertRaisesRegex(ValueError,"Wrong/changed"):
                app.load_bundle(tmp)


if __name__ == "__main__":
    unittest.main(verbosity=2)
