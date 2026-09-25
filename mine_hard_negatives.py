#!/usr/bin/env python3
"""Mine training hard negatives. Output: ONE CSV and 30 review images.

1. Download the frozen 640 best.pt from the private Hugging Face backup.
2. Score audited training negatives: hardness = highest fracture confidence.
3. Rank images and propose a capped candidate pool. Review before training.
No training, relabeling, or validation/test image inference occurs here.
"""
import argparse
import csv
import hashlib
import json
import tempfile
from collections import Counter
from pathlib import Path

import cv2
import torch
import ultralytics
from huggingface_hub import hf_hub_download
from tqdm import tqdm
from ultralytics import YOLO

# Fixed checkpoint provenance and the same proposal rule as the previous script.
REPO = "Crimson-Dawn/grazpedwri-yolo26-checkpoints"
REVISION = "d104d0498236676d0882942a37ee1c353d80cfda"
RUN = "baseline_yolo26s_640_100epochs_seed42_from50_20260917_163013_311680Z"
REMOTE = f"runs/{RUN}/epoch_100"
MANIFEST_SHA = "1719f37f442512c3c4fcab8350bedc0f27ff2dcebc082761299912b897747034"
MIN_SCORE, MAX_FRACTION, MAX_PER_PATIENT = 0.10, 0.20, 2
REVIEW_IMAGES = 30


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def download_checkpoint():
    # Uses cached HF login or HF_TOKEN; never put a token in this script.
    metadata_path = hf_hub_download(REPO, f"{REMOTE}/checkpoint_info.json", revision=REVISION)
    metadata = json.loads(Path(metadata_path).read_text())
    assert metadata["manifest_sha256"] == MANIFEST_SHA, "Backup uses a different split."
    assert metadata["completed_epoch"] == 100 and metadata["run_name"] == RUN
    checkpoint = hf_hub_download(REPO, f"{REMOTE}/best.pt", revision=REVISION)
    # HF stores the pre-stripping training checkpoint. Its container checksum may
    # differ from the local post-training best.pt, so verify against its backup.
    assert sha256(checkpoint) == metadata["best_pt_sha256"], "Checkpoint checksum mismatch."
    return checkpoint


def save_preview(row, destination):
    image = cv2.imread(row["image_path"])
    assert image is not None, f"Cannot read {row['image_path']}"
    if row["box"] is not None:
        x1, y1, x2, y2 = map(round, row["box"])
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    # Keep original resolution. Only the highest-confidence prediction is drawn.
    image = cv2.copyMakeBorder(image, 65, 0, 0, max(0, 900 - image.shape[1]),
                               cv2.BORDER_CONSTANT, value=(255, 255, 255))
    lines = [f"Rank {row['rank']} | {row['image_id']}",
             f"TRAIN negative | model score {row['hardness_score']:.4f} | candidate: {row['candidate']}"]
    for index, text in enumerate(lines):
        cv2.putText(image, text, (10, 24 + index * 27), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 0), 1, cv2.LINE_AA)
    assert cv2.imwrite(str(destination), image), f"Could not write {destination}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/grazpedwri_yolo"))
    parser.add_argument("--output", type=Path, default=Path("hard_negatives_review"))
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()
    data, output = args.data.resolve(), args.output.resolve()
    assert not output.exists(), "Choose a new --output folder; existing results are never overwritten."
    assert not output.is_relative_to(data), "Keep output outside the prepared dataset."
    assert args.batch > 0 and torch.cuda.is_available(), "A GPU and positive batch size are required."
    assert ultralytics.__version__ == "8.4.152", "Use the existing ultralytics==8.4.152 YOLO environment."

    # 1. Read only audited TRAIN negatives from the unchanged frozen manifest.
    manifest = data / "split_manifest.csv"
    assert sha256(manifest) == MANIFEST_SHA, "The frozen manifest has changed."
    with manifest.open(newline="", encoding="utf-8-sig") as handle:
        negatives = [r for r in csv.DictReader(handle)
                     if r["split"] == "train" and r["sample_type"] == "negative"]
    assert len(negatives) == 3961, "Expected 3,961 audited training negatives."
    images = {}
    for row in negatives:
        stem = row["filestem"]
        image = data / "images/train" / f"{stem}{Path(row['image_relpath']).suffix}"
        label = data / "labels/train" / f"{stem}.txt"
        assert int(row["fracture_count"]) == 0 and image.is_file()
        assert not label.read_text().strip(), f"Nonempty fracture label on negative image: {stem}"
        images[stem] = (str(image), int(row["patient_id"]))

    # 2. Automatically download the 640 baseline; previous local runs are unnecessary.
    model = YOLO(download_checkpoint())
    assert len(model.names) == 1 and str(model.names[0]).lower() == "fracture"
    results = []
    # A temporary input list streams files without loading all X-rays into RAM.
    # It is removed automatically and is not another output file to manage.
    with tempfile.TemporaryDirectory() as temp:
        source = Path(temp) / "images.txt"
        source.write_text("\n".join(images[k][0] for k in sorted(images)))
        predictions = model.predict(source=str(source), stream=True, batch=args.batch,
                                    device=0, imgsz=640, conf=0.001, iou=0.70, max_det=300,
                                    quantize=16, rect=False, nms=None, save=False, verbose=False)
        for prediction in tqdm(predictions, total=len(images), desc="Scoring training negatives"):
            stem = Path(prediction.path).stem
            scores = prediction.boxes.conf.cpu()
            assert torch.isfinite(scores).all(), "Invalid prediction scores."
            best = int(scores.argmax()) if len(scores) else None
            results.append({"image_id": stem, "patient_id": images[stem][1],
                            "image_path": images[stem][0],
                            "hardness_score": float(scores[best]) if best is not None else 0.0,
                            "box": prediction.boxes.xyxy[best].cpu().tolist() if best is not None else None})
    assert len(results) == len(images) and {r["image_id"] for r in results} == set(images)

    # 3. Rank by score. Propose at most 20%, score >= 0.10, at most two per patient.
    results.sort(key=lambda r: (-r["hardness_score"], r["image_id"]))
    selected, patient_counts = 0, Counter()
    limit = int(len(results) * MAX_FRACTION)
    for rank, row in enumerate(results, 1):
        candidate = (row["hardness_score"] >= MIN_SCORE and selected < limit
                     and patient_counts[row["patient_id"]] < MAX_PER_PATIENT)
        row.update(rank=rank, candidate="yes" if candidate else "no", review="", notes="")
        if candidate:
            selected += 1
            patient_counts[row["patient_id"]] += 1

    # 4. Save ONE CSV. All negatives remain visible; candidate=yes is provisional.
    output.mkdir(parents=True)
    fields = ["rank", "image_id", "patient_id", "hardness_score", "candidate", "review", "notes", "image_path"]
    with (output / "ranked_negatives.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({**row, "hardness_score": f"{row['hardness_score']:.6f}"} for row in results)

    # 5. Save the highest-ranked images for human review, not new training labels.
    previews = output / "review_images"
    previews.mkdir()
    for row in results[:REVIEW_IMAGES]:
        save_preview(row, previews / f"{row['rank']:03d}_{row['image_id']}.png")
    print(f"\nDone: {len(results)} images ranked; {selected} provisional candidates (cap {limit}).")
    print(f"Output: {output}")
    print("Review the images and use CSV review/notes columns to flag keep/exclude/uncertain.")
    print("No labels changed, no training started, and no final training pool frozen.")


if __name__ == "__main__":
    main()
