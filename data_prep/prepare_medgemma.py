"""Prepare MedGemma localization data from an already-built frozen YOLO dataset.

No downloads, model loading, GPU use, new split, or source-file modifications.
Run this script from your project (or pass --root /path/to/project).

Prompt provenance: adapted from Google's PUBLIC INFERENCE/LOCALIZATION NOTEBOOK,
not claimed to reproduce an undisclosed internal training prompt.
The public notebook uses [y0,x0,y1,x1] in [0,1000] and a Final Answer JSON list.
We adapt the query to all visible wrist fractures, add empty-list negatives,
and omit chest-specific laterality hints and requests for unsupported reasoning.

Data format follows the official fine-tuning notebook: a separate image field,
a user message with an image placeholder and instruction, and an assistant target.
Validation/test generation inputs are saved separately, WITHOUT assistant answers.
"""
from pathlib import Path
from collections import Counter
from contextlib import ExitStack
import argparse
import hashlib
import io
import json
import math
import os
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from config import DATASET_SOURCE, MANIFEST_SHA256, SPLIT_COUNTS, load_config, resolve_path, sha256

ROOT = PROJECT_ROOT
DATASET = DATASET_SOURCE
MODEL_ID = "google/medgemma-1.5-4b-it"
SPLITS = ["train", "val", "test"]
SAMPLE_TYPES = ["positive", "negative", "excluded"]
EXPECTED_SPLIT_COUNTS = SPLIT_COUNTS
COORDINATE_DECIMALS = 2  # 0.01 on the 1000-scale; avoids coarse integer rounding.
BOUNDS_TOLERANCE = 1e-6  # Only accommodate tiny rounding drift at image boundaries.
REFERENCE_COMMIT = "a60a66024f6153496dfa9490dba12d4f1fbd092e"
LOCALIZATION_REFERENCE = (
    "https://github.com/Google-Health/medgemma/blob/"
    + REFERENCE_COMMIT
    + "/notebooks/cxr_anatomy_localization_with_hugging_face.ipynb"
)
FINETUNING_REFERENCE = (
    "https://github.com/Google-Health/medgemma/blob/"
    + REFERENCE_COMMIT
    + "/notebooks/fine_tune_with_hugging_face.ipynb"
)

PROMPT = '''Instructions:
The following user query requires outputting bounding boxes. The format of bounding box coordinates is [y0, x0, y1, x1], where (y0, x0) is the top-left corner and (y1, x1) is the bottom-right corner. This implies x0 < x1 and y0 < y1.
Always normalize the x and y coordinates to the range [0, 1000], relative to the entire image. A position at 15% of the image width corresponds to an x coordinate of 150. Decimal coordinates are allowed.
Return one box for each visible fracture. Each object must contain a "box_2d" coordinate array and a "label" whose value is "fracture". Order the boxes by top coordinate, then left coordinate.
If no visible fracture is present, return an empty JSON list []. Do not return boxes for text markers, casts, metal implants, or normal anatomy.
Output only "Final Answer: " followed by a single parseable JSON list enclosed in a json code fence. Do not add reasoning or other commentary.

Query:
Where are all the visible fractures, if any, in this wrist X-ray?
Answer:'''


def require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def write_jsonl(handle, value):
    handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def convert_boxes(boxes, stem):
    """YOLO [0,cx,cy,w,h] -> [{box_2d:[top,left,bottom,right],label:fracture}]."""
    require(isinstance(boxes, list), f"Boxes must be a list: {stem}")
    converted = []
    clamped_boxes = 0
    max_rounding_error = 0.0
    for box in boxes:
        require(isinstance(box, list) and len(box) == 5, f"Invalid box: {stem}")
        require(all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in box),
                f"Non-numeric box: {stem}")
        require(all(math.isfinite(v) for v in box), f"Non-finite box: {stem}")
        require(box[0] == 0, f"Expected already-converted fracture class 0: {stem}")
        cx, cy, w, h = map(float, box[1:])
        require(0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1,
                f"Invalid normalized YOLO coordinates: {stem}")
        corners = [cy - h / 2, cx - w / 2, cy + h / 2, cx + w / 2]
        require(all(-BOUNDS_TOLERANCE <= v <= 1 + BOUNDS_TOLERANCE for v in corners),
                f"Box extends materially outside the image; review rather than silently alter it: {stem}: {corners}")
        clipped = [min(1.0, max(0.0, v)) for v in corners]
        clamped_boxes += int(clipped != corners)
        target = [round(v * 1000, COORDINATE_DECIMALS) for v in clipped]
        require(target[0] < target[2] and target[1] < target[3],
                f"Box collapsed after coordinate formatting: {stem}")
        max_rounding_error = max(max_rounding_error, max(
            abs(a / 1000 - b) for a, b in zip(target, clipped)
        ))
        converted.append({"box_2d": target, "label": "fracture"})
    converted.sort(key=lambda obj: tuple(obj["box_2d"]))
    return converted, clamped_boxes, max_rounding_error


def target_text(boxes):
    return "Final Answer: ```json\n" + json.dumps(
        boxes, separators=(",", ":"), allow_nan=False
    ) + "\n```"


def user_message():
    # Image content is supplied separately as a PIL image by the training collator.
    # No diagnoses, AO classifications, sample types, or patient metadata in the prompt.
    return {
        "role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": PROMPT}],
    }


def model_record(image_id, patient_id, split, image, answer=None):
    messages = [user_message()]
    if answer is not None:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
    return {"image_id": image_id, "patient_id": patient_id, "split": split,
            "image": image, "messages": messages}


def build(config):
    """Build from a ProjectConfig after validating frozen provenance."""
    if not hasattr(config, "project_root"):
        config = load_config(project_root=config)
    root = config.project_root
    source = config.yolo_data_dir
    manifest_path = source / "split_manifest.csv"
    source_info_path = source / "dataset_info.json"
    output = config.medgemma_data_dir
    staging = output.with_name(output.name + ".building")

    if os.path.lexists(str(output)) or os.path.lexists(str(staging)):
        raise FileExistsError(
            f"{output} or {staging} already exists. Nothing is overwritten. "
            "Use the completed data or inspect the unfinished build."
        )
    if not manifest_path.is_file() or not source_info_path.is_file():
        raise FileNotFoundError("Run your YOLO dataset rebuild script first: " + str(source))

    manifest_bytes = manifest_path.read_bytes()
    manifest_hash = sha256_bytes(manifest_bytes)
    require(manifest_hash == MANIFEST_SHA256,
            f"Frozen manifest SHA256 mismatch: expected {MANIFEST_SHA256}, got {manifest_hash}")
    source_info = json.loads(source_info_path.read_text(encoding="utf-8"))
    require(source_info.get("dataset") == DATASET, "Unexpected source dataset version.")
    require(source_info.get("manifest_sha256") == manifest_hash,
            "YOLO manifest differs from the manifest used to build that dataset.")
    root_manifest = config.manifest_path
    if not root_manifest.is_file():
        raise FileNotFoundError(
            "Configured root manifest is required: " + str(root_manifest)
        )
    root_manifest_bytes = root_manifest.read_bytes()
    root_manifest_hash = sha256_bytes(root_manifest_bytes)
    require(
        root_manifest_hash == MANIFEST_SHA256,
        f"Configured root manifest SHA256 mismatch: expected {MANIFEST_SHA256}, got {root_manifest_hash}",
    )
    require(
        root_manifest_bytes == manifest_bytes,
        "Project-root and prepared-YOLO manifests differ. Resolve this before proceeding.",
    )
    dataset_root = Path(source_info["dataset_root"]).resolve(strict=True)

    # Execution-only imports keep --help/--print-config free of image and data packages.
    import cv2
    import numpy as np
    import pandas as pd
    from PIL import Image

    m = pd.read_csv(io.BytesIO(manifest_bytes), dtype={"patient_id": "string", "filestem": "string"})
    required = {"patient_id", "filestem", "split", "sample_type", "image_relpath",
                "fracture_labels_json", "fracture_count"}
    require(required <= set(m.columns), f"Missing columns: {required - set(m.columns)}")
    require(m[list(required)].notna().all().all(), "Missing required manifest values.")
    require(m["filestem"].is_unique, "Duplicate image IDs.")
    require(m["patient_id"].str.fullmatch(r"[0-9]+").all(), "Expected numeric anonymous GRAZ patient IDs.")
    require(m["split"].isin(SPLITS).all(), "Missing/invalid frozen patient assignments.")
    require(m["sample_type"].isin(SAMPLE_TYPES).all(), "Unknown sample category.")
    patient_keys = m["patient_id"].map(lambda value: str(int(value)))
    require(m.assign(_patient=patient_keys).groupby("_patient")["split"].nunique().eq(1).all(),
            "Patient leakage, including alternate zero-padded IDs.")

    counts = pd.crosstab(m["split"], m["sample_type"]).reindex(
        index=SPLITS, columns=SAMPLE_TYPES, fill_value=0
    )
    expected = pd.DataFrame.from_dict(EXPECTED_SPLIT_COUNTS, orient="index").reindex(
        index=SPLITS, columns=SAMPLE_TYPES
    )
    require(np.array_equal(counts.to_numpy(), expected.to_numpy()),
            "Counts differ from the frozen audited baseline: " + str(counts.to_dict(orient="index")))

    # Validate every included source and label before creating any output directory.
    included = m[m["sample_type"].isin(["positive", "negative"])].sort_values(["split", "filestem"])
    prepared = []
    total_clamps, max_rounding_error = 0, 0.0
    for row in included.itertuples(index=False):
        stem = str(row.filestem)
        require(Path(stem).name == stem and stem not in {"", ".", ".."}, f"Invalid image ID: {stem}")
        require(stem.split("_")[0].isdigit() and int(stem.split("_")[0]) == int(row.patient_id),
                f"Filename/patient mismatch: {stem}")
        original_relative = Path(row.image_relpath)
        require(not original_relative.is_absolute(), f"Absolute source path in manifest: {stem}")
        expected_image = (dataset_root / original_relative).resolve(strict=True)
        require(expected_image.is_relative_to(dataset_root), f"Source path escapes the dataset: {stem}")
        image = source / "images" / row.split / f"{stem}{original_relative.suffix}"
        label = source / "labels" / row.split / f"{stem}.txt"
        require(image.is_file() and label.is_file(), f"Missing YOLO image/label pair: {stem}")
        require(image.resolve() == expected_image and expected_image.stem == stem,
                f"YOLO image link does not match the manifest source: {stem}")

        boxes = json.loads(row.fracture_labels_json)
        converted, clamps, rounding_error = convert_boxes(boxes, stem)
        require(len(boxes) == row.fracture_count, f"Box count mismatch: {stem}")
        require(bool(boxes) == (row.sample_type == "positive"), f"Sample category/box mismatch: {stem}")
        label_rows = [list(map(float, line.split())) for line in label.read_text().splitlines() if line.strip()]
        require(len(label_rows) == len(boxes), f"Prepared label count differs from manifest: {stem}")
        for actual, original in zip(label_rows, boxes):
            require(len(actual) == 5 and all(math.isfinite(v) for v in actual), f"Invalid YOLO label: {stem}")
            require(all(math.isclose(a, b, abs_tol=1e-9, rel_tol=0) for a, b in zip(actual, original)),
                    f"YOLO label differs from frozen manifest: {stem}")
        total_clamps += clamps
        max_rounding_error = max(max_rounding_error, rounding_error)
        prepared.append((row, image, boxes, converted))

    staging.mkdir(parents=True, exist_ok=False)
    for split in SPLITS:
        (staging / "images" / split).mkdir(parents=True)
    (staging / "references").mkdir()
    (staging / "split_manifest.csv").write_bytes(manifest_bytes)
    (staging / "prompt.txt").write_text(PROMPT, encoding="utf-8")
    metadata_rows = []
    modes = Counter()
    image_bytes = 0

    with ExitStack() as stack:
        sft_files = {s: stack.enter_context((staging / f"{s}.jsonl").open("x", encoding="utf-8"))
                     for s in ["train", "val"]}
        input_files = {s: stack.enter_context((staging / f"{s}_inputs.jsonl").open("x", encoding="utf-8"))
                       for s in ["val", "test"]}
        reference_files = {s: stack.enter_context((staging / "references" / f"{s}.jsonl").open("x", encoding="utf-8"))
                           for s in SPLITS}

        for index, (row, image, original_boxes, converted_boxes) in enumerate(prepared, start=1):
            stem = str(row.filestem)
            relative_image = (Path("images") / row.split / f"{stem}.png").as_posix()
            destination = staging / relative_image
            # Reading a header here does not convert or clip pixel values.
            with Image.open(image) as header:
                require(header.format == "PNG", f"Unexpected image format: {stem}")
                width, height = header.size
                source_mode = header.mode
                require(header.getexif().get(274, 1) in (None, 1),
                        f"Unexpected EXIF orientation; coordinate mapping needs review: {stem}")
            # IMPORTANT: do not use PIL .convert('RGB') directly on raw 16-bit data.
            # OpenCV IMREAD_COLOR performs the same standard 8-bit decoding used
            # by the YOLO image loader, then we convert channel order to RGB.
            bgr = cv2.imread(str(image), cv2.IMREAD_COLOR)
            require(bgr is not None, f"Image decode failed: {stem}")
            require(bgr.dtype == np.uint8 and bgr.ndim == 3 and bgr.shape[2] == 3,
                    f"Unexpected decoded image type: {stem}")
            require(bgr.shape[:2] == (height, width), f"Unexpected geometry change: {stem}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            Image.fromarray(rgb).save(destination, format="PNG", compress_level=3)
            modes[source_mode] += 1
            image_bytes += destination.stat().st_size

            answer = target_text(converted_boxes)
            basic = {"image_id": stem, "patient_id": str(row.patient_id), "split": row.split,
                     "image": relative_image}
            if row.split in sft_files:
                write_jsonl(sft_files[row.split], model_record(**basic, answer=answer))
            if row.split in input_files:
                # Use THESE files for generation evaluation. No target answer or
                # answer-revealing sample category is included in these records.
                write_jsonl(input_files[row.split], model_record(**basic))
            write_jsonl(reference_files[row.split], {
                **basic, "width": width, "height": height, "sample_type": row.sample_type,
                "original_yolo_boxes": original_boxes, "box_2d_targets": converted_boxes,
                "expected_answer": answer,
            })
            metadata_rows.append({
                "filestem": stem, "patient_id": str(row.patient_id), "split": row.split,
                "sample_type": row.sample_type, "medgemma_image_relpath": relative_image,
                "width": width, "height": height, "source_mode": source_mode,
                "fracture_count": len(original_boxes),
                "box_2d_targets_json": json.dumps(converted_boxes, allow_nan=False),
            })
            if index % 500 == 0 or index == len(prepared):
                print(f"Prepared {index:,}/{len(prepared):,} images", flush=True)

    pd.DataFrame(metadata_rows).to_csv(staging / "medgemma_manifest.csv", index=False)
    actual_counts = Counter(r[0].split for r in prepared)
    require(all(actual_counts[s] == int(counts.loc[s, "positive"] + counts.loc[s, "negative"]) for s in SPLITS),
            "Unexpected output sample counts.")
    write_json(staging / "dataset_info.json", {
        "dataset": DATASET, "model_id": MODEL_ID, "source_yolo_directory": str(source),
        "shared_config": config.public_dict(),
        "manifest_sha256": manifest_hash, "prompt_sha256": sha256_bytes(PROMPT.encode()),
        "preparation_script_sha256": sha256_bytes(Path(__file__).read_bytes()),
        "included_images": len(prepared), "split_counts": counts.to_dict(orient="index"),
        "patients_assigned": m.assign(_patient=patient_keys).groupby("split")["_patient"].nunique().to_dict(),
        "patients_included": included.groupby("split")["patient_id"].nunique().to_dict(),
        "fracture_boxes": int(included["fracture_count"].sum()),
        "uncertain_boxed_positives_retained": int((included["sample_type"].eq("positive") & included["diagnosis_uncertain"].eq(1)).sum()) if "diagnosis_uncertain" in included else None,
        "image_preparation": "OpenCV IMREAD_COLOR 8-bit decoding, BGR to RGB, lossless RGB PNG; same native dimensions; no crop, padding, resize, contrast changes or augmentation",
        "source_image_modes": dict(modes), "converted_image_bytes": image_bytes,
        "coordinate_format": "[y0,x0,y1,x1], normalized to [0,1000] against the entire original image",
        "coordinate_decimals": COORDINATE_DECIMALS,
        "max_coordinate_rounding_error_normalized": max_rounding_error,
        "boxes_with_tiny_boundary_drift_clamped": total_clamps,
        "boundary_tolerance_normalized": BOUNDS_TOLERANCE,
        "box_order": "top, left, bottom, right",
        "prompt_reference": LOCALIZATION_REFERENCE,
        "fine_tuning_format_reference": FINETUNING_REFERENCE,
        "prompt_provenance": "Adapted public localization inference example, not a verified copy of Google's internal training prompt",
        "prompt_changes": ["Wrist fractures instead of chest anatomy", "All visible fractures, with explicit empty-list negatives", "No chest-specific left/right hints", "No requested reasoning because no expert reasoning targets are available", "Decimal coordinates and deterministic box order"],
        "label_policy": "Preserve existing positives/negatives/exclusions, including boxed uncertain positives; no new confidence labels",
        "generation_evaluation": "Use val_inputs.jsonl or test_inputs.jsonl; never provide assistant targets or reference records to the model",
        "software": {"python": sys.version, "pandas": pd.__version__, "numpy": np.__version__, "opencv": cv2.__version__, "pillow": Image.__version__},
    })
    write_json(staging / "processor_requirements.json", {
        "model_id": MODEL_ID,
        "apply_model_processor": True,
        "do_pan_and_scan": False,
        "geometry_policy": "Keep full image. Processor full-image resizing/normalization is allowed. No untracked cropping, rotation or padding.",
        "image_mode": "RGB", "image_range_on_disk": "uint8 [0,255]",
        "coordinate_frame": "Entire original image, normalized to [0,1000]",
        "training_note": "This file documents settings to apply in the later training/inference code; the builder does not load a processor or model.",
    })
    if os.path.lexists(str(output)):
        raise FileExistsError(f"Output appeared during the build; refusing to overwrite: {output}")
    staging.rename(output)

    summary = counts.copy()
    summary.insert(0, "patients_assigned", m.assign(_patient=patient_keys).groupby("split")["_patient"].nunique())
    summary.insert(1, "patients_included", included.groupby("split")["patient_id"].nunique())
    print("\n" + summary.to_string())
    print("\nPatient overlap: NONE")
    print("Frozen manifest SHA256:", manifest_hash)
    print("Converted image storage:", round(image_bytes / 1024**3, 2), "GiB")
    print("MedGemma data ready:", output)
    print("Train with train.jsonl; use val.jsonl for validation loss.")
    print("Generate predictions from val_inputs.jsonl/test_inputs.jsonl, which contain NO answers.")
    print("Original images, YOLO labels and frozen split are unchanged.")
    return output


def _resolve_dataset_images(dataset, data_root):
    from datasets import Image as DatasetImage
    data_root = Path(data_root).resolve()

    def resolve_image(example):
        image = (data_root / example["image"]).resolve(strict=True)
        require(image.is_relative_to(data_root), "Image path escapes prepared MedGemma directory.")
        return {"image": str(image)}

    return dataset.map(resolve_image).cast_column("image", DatasetImage())


def _resolve_medgemma_data_root(data_root):
    config = load_config()
    config.activate_environment()
    if data_root is None:
        return config.medgemma_data_dir
    return resolve_path(data_root, config.project_root)


def load_sft_datasets(data_root=None):
    """Optional helper for the later training script: PIL images + official-style messages.

    Returns only train and validation. Test is never returned to the trainer.
    Requires the Hugging Face `datasets` package when this function is called.
    """
    data_root = _resolve_medgemma_data_root(data_root)
    from datasets import load_dataset
    datasets = load_dataset("json", data_files={
        "train": str(data_root / "train.jsonl"),
        "validation": str(data_root / "val.jsonl"),
    })
    return _resolve_dataset_images(datasets, data_root)


def load_inference_dataset(split="val", data_root=None):
    """Load evaluation inputs with image placeholders, but NO assistant answers."""
    require(split in {"val", "test"}, "Choose val or test for held-out generation.")
    data_root = _resolve_medgemma_data_root(data_root)
    from datasets import load_dataset
    dataset = load_dataset("json", data_files={split: str(data_root / f"{split}_inputs.jsonl")})[split]
    return _resolve_dataset_images(dataset, data_root)


def _config_with_overrides(args):
    config = load_config(project_root=args.project_root, env_file=args.env_file)
    if args.data_dir is not None or args.manifest_path is not None:
        from dataclasses import replace
        config = replace(
            config,
            data_dir=resolve_path(args.data_dir, config.project_root) if args.data_dir else config.data_dir,
            manifest_path=resolve_path(args.manifest_path, config.project_root) if args.manifest_path else config.manifest_path,
        )
    return config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", "--root", dest="project_root", type=Path,
                        default=PROJECT_ROOT, help="Repository root containing prepared YOLO data")
    parser.add_argument("--env-file", type=Path, help="Optional .env file, relative to project root")
    parser.add_argument("--data-dir", type=Path, help="Override configured data directory")
    parser.add_argument("--manifest-path", type=Path, help="Override configured frozen manifest")
    parser.add_argument("--print-config", action="store_true", help="Print resolved non-secret configuration and exit")
    args = parser.parse_args(argv)
    config = _config_with_overrides(args)
    if args.print_config:
        print(json.dumps(config.public_dict(), indent=2, sort_keys=True))
        return
    build(config)


if __name__ == "__main__":
    main()
