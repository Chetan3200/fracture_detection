"""Run fine-tuned MedGemma fracture inference on one image or an image folder.

Examples:
  ./.venv/bin/python fracture_evaluation/infer_medgemma.py --checkpoint runs/MY_RUN \
      --source radiographs/ --protocol validation/evaluation.json
  ./.venv/bin/python fracture_evaluation/infer_medgemma.py --hf --hf-run MY_RUN \
      --source radiograph.png --conf 0.25 --output inference_output

This inference-only CLI writes JSONL predictions and, unless disabled, annotated
images. It does not load ground truth, datasets, or compute AP metrics.
"""
from __future__ import annotations

import argparse
import gc
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from evaluation_scripts import checkpoints
from evaluation_scripts import medgemma_model
import inference_common as ic


def arguments():
    """Parse the inference-only MedGemma command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    ic.add_arguments(parser, "medgemma")
    parser.add_argument(
        "--hf-run",
        help="Run directory name in HF; auto-select only when one completed full run exists",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Defaults to the saved training value unless frozen by the protocol",
    )
    return parser.parse_args()


def main():
    args = arguments()
    ic.validate_args(args, "medgemma")

    report = ic.read_protocol(args.protocol, "medgemma")
    cutoff = ic.resolve_cutoff(args, report)
    max_new_tokens = ic.option_from_protocol(
        args.max_new_tokens, report, ("max_new_tokens",), None
    )
    files, source_root, output = ic.plan_inputs(
        args.source, args.recursive, args.output, "medgemma"
    )

    # Keep heavyweight runtime dependencies out of module import so --help and
    # argument/protocol validation do not require the MedGemma environment.
    import cv2
    import torch
    from PIL import Image

    if not torch.cuda.is_available() or args.device >= torch.cuda.device_count():
        raise ValueError("Requested CUDA device unavailable.")
    torch.cuda.set_device(args.device)

    predictor = None
    try:
        run, adapter, provenance = checkpoints.resolve_medgemma(
            local=args.checkpoint,
            repo=args.hf_repo,
            run=args.hf_run,
            revision=args.hf_revision,
            cache_dir=args.cache_dir,
        )
        predictor = medgemma_model.MedGemmaPredictor(
            run, adapter, args.device, max_new_tokens
        )
        settings = {
            **predictor.settings,
            "opencv": cv2.__version__,
            "source_image_decode": (
                "OpenCV IMREAD_COLOR uint8 BGR -> RGB; native dimensions, no other change"
            ),
        }
        settings.pop("device_index", None)
        ic.verify_protocol(
            report,
            "medgemma",
            provenance,
            settings,
            {"torch": str(torch.__version__)},
            {
                "checkpoints.py": checkpoints.__file__,
                "medgemma_model.py": medgemma_model.__file__,
            },
        )

        with ic.InferenceWriter(
            output,
            "medgemma",
            source_root,
            files,
            cutoff,
            provenance,
            settings,
            args.protocol,
            not args.no_annotate,
            code_paths=[
                __file__,
                ic.__file__,
                checkpoints.__file__,
                medgemma_model.__file__,
            ],
        ) as writer:
            for index, file in enumerate(files, start=1):
                bgr = ic.read_bgr(file)
                image = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                try:
                    prediction = predictor.predict(image)
                finally:
                    image.close()
                record = writer.add(file, bgr, prediction)
                print(
                    f"{index}/{len(files)} {file}: {record['status']}, "
                    f"{len(record['detections'])} boxes",
                    flush=True,
                )
        print(f"Inference output: {output}", flush=True)
    finally:
        if predictor is not None:
            del predictor
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
