"""Rebuild the YOLO dataset from the existing frozen manifest. No new split."""
from pathlib import Path
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
from config import (
    DATASET_SOURCE,
    MANIFEST_SHA256,
    SPLIT_COUNTS,
    load_config,
    resolve_path,
)

SPLITS = ["train", "val", "test"]


def require(condition, message):
    if not condition:
        raise ValueError(message)



def _lexists(path):
    return os.path.lexists(str(path))


def _config_with_overrides(args):
    config = load_config(project_root=args.project_root, env_file=args.env_file)
    if (
        args.data_dir is not None
        or args.manifest_path is not None
        or args.cache_dir is not None
        or args.kagglehub_cache is not None
    ):
        from dataclasses import replace

        config = replace(
            config,
            data_dir=(
                resolve_path(args.data_dir, config.project_root)
                if args.data_dir
                else config.data_dir
            ),
            manifest_path=(
                resolve_path(args.manifest_path, config.project_root)
                if args.manifest_path
                else config.manifest_path
            ),
            cache_dir=(
                resolve_path(args.cache_dir, config.project_root)
                if args.cache_dir
                else config.cache_dir
            ),
            kagglehub_cache=(
                resolve_path(args.kagglehub_cache, config.project_root)
                if args.kagglehub_cache
                else config.kagglehub_cache
            ),
        )
    return config


def build(config):
    """Build from a ProjectConfig after validating all local frozen inputs."""
    if not hasattr(config, "project_root"):
        config = load_config(project_root=config)
    manifest = config.manifest_path
    output = config.yolo_data_dir
    staging = output.with_name(output.name + ".building")
    if not manifest.is_file():
        raise FileNotFoundError(f"Copy your original frozen manifest to {manifest}")
    if _lexists(output) or _lexists(staging):
        raise FileExistsError(
            f"{output} or {staging} already exists. "
            "Use the completed dataset, or inspect the unfinished build; "
            "nothing is overwritten."
        )

    manifest_bytes = manifest.read_bytes()
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    if manifest_hash != MANIFEST_SHA256:
        raise ValueError(
            "Frozen manifest SHA256 mismatch: "
            f"expected {MANIFEST_SHA256}, got {manifest_hash}"
        )

    # Execution-only imports keep --help/--print-config free of data/ML dependencies.
    import pandas as pd
    m = pd.read_csv(
        io.BytesIO(manifest_bytes),
        dtype={"patient_id": "string", "filestem": "string"},
    )
    required = {
        "patient_id",
        "filestem",
        "split",
        "sample_type",
        "image_relpath",
        "fracture_labels_json",
        "fracture_count",
    }
    require(required <= set(m.columns), f"Missing columns: {required - set(m.columns)}")
    require(m[list(required)].notna().all().all(), "Missing manifest values.")
    require(m["filestem"].is_unique, "Duplicate image IDs.")
    require(m["patient_id"].str.strip().ne("").all(), "Empty patient IDs.")
    require(m["split"].isin(SPLITS).all(), "Missing or invalid frozen split assignments.")
    require(m.groupby("patient_id")["split"].nunique().eq(1).all(), "Patient leakage!")
    require(m["sample_type"].value_counts().to_dict() == {
        "positive": 13550,
        "negative": 5657,
        "excluded": 1120,
    }, "Manifest does not match the audited dataset.")
    counts = pd.crosstab(m["split"], m["sample_type"]).reindex(
        index=SPLITS,
        columns=["positive", "negative", "excluded"],
        fill_value=0,
    )
    require(counts.to_dict(orient="index") == SPLIT_COUNTS, "Split counts differ from the frozen baseline.")

    config.activate_environment()
    cache = config.kagglehub_cache or (config.cache_dir / "kagglehub")
    os.environ["KAGGLEHUB_CACHE"] = str(cache)
    import kagglehub
    dataset_root = Path(kagglehub.dataset_download(DATASET_SOURCE)).resolve()

    prepared = []
    included = m[m["sample_type"].isin(["positive", "negative"])]
    for row in included.itertuples(index=False):
        image = (dataset_root / row.image_relpath).resolve()
        require(image.is_relative_to(dataset_root), f"Invalid image path: {image}")
        require(image.is_file(), f"Missing image: {image}")
        require(
            Path(row.filestem).name == row.filestem and row.filestem not in {".", ".."},
            f"Invalid filestem: {row.filestem}",
        )

        boxes = json.loads(row.fracture_labels_json)
        require(len(boxes) == row.fracture_count, f"Box count mismatch: {row.filestem}")
        require(bool(boxes) == (row.sample_type == "positive"), row.filestem)

        for box in boxes:
            require(len(box) == 5 and box[0] == 0, f"Invalid class/box: {row.filestem}")
            require(all(math.isfinite(float(v)) for v in box), row.filestem)

            x, y, w, h = map(float, box[1:])
            require(
                0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1,
                row.filestem,
            )
        prepared.append((row, image, boxes))

    # Reserve the staging directory itself, not just its children.
    staging.mkdir(parents=True, exist_ok=False)
    for split in SPLITS:
        (staging / "images" / split).mkdir(parents=True)
        (staging / "labels" / split).mkdir(parents=True)
    for row, image, boxes in prepared:
        destination = staging / "images" / row.split / f"{row.filestem}{image.suffix}"
        destination.symlink_to(image)
        label = staging / "labels" / row.split / f"{row.filestem}.txt"
        lines = [" ".join(map(str, box)) for box in boxes]
        label.write_text("\n".join(lines) + ("\n" if lines else ""))
    (staging / "data.yaml").write_text(
        f"path: {json.dumps(str(output))}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "nc: 1\n"
        "names:\n"
        "  0: fracture\n"
    )
    (staging / "split_manifest.csv").write_bytes(manifest_bytes)

    info = {
        "dataset": DATASET_SOURCE,
        "dataset_root": str(dataset_root),
        "manifest_sha256": manifest_hash,
        "included_images": len(prepared),
        "split_counts": counts.to_dict(orient="index"),
        "label_policy": (
            "Reuse manifest class-0 boxes; include positives/negatives only."
        ),
        "shared_config": config.public_dict(),
    }
    (staging / "dataset_info.json").write_text(json.dumps(info, indent=2))
    if _lexists(output):
        raise FileExistsError(f"Output appeared during the build; refusing to overwrite: {output}")
    staging.rename(output)
    counts.insert(
        0,
        "patients_assigned",
        m.groupby("split")["patient_id"].nunique(),
    )
    print(counts.to_string())
    print("\nPatient overlap: NONE")
    print("Manifest SHA256:", manifest_hash)
    print("Dataset ready:", output / "data.yaml")
    print("Do not delete", cache, ": the image symlinks depend on it.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT, help="Repository root (default: script parent)")
    parser.add_argument("--env-file", type=Path, help="Optional .env file, relative to project root")
    parser.add_argument("--data-dir", type=Path, help="Override configured data directory")
    parser.add_argument("--manifest-path", type=Path, help="Override configured frozen manifest")
    parser.add_argument("--cache-dir", type=Path, help="Override configured cache directory")
    parser.add_argument("--kagglehub-cache", type=Path, help="Override configured KaggleHub cache")
    parser.add_argument("--print-config", action="store_true", help="Print resolved non-secret configuration and exit")
    args = parser.parse_args(argv)
    config = _config_with_overrides(args)
    if args.print_config:
        print(json.dumps(config.public_dict(), indent=2, sort_keys=True))
        return
    build(config)


if __name__ == "__main__":
    main()
