"""Evaluate YOLO26: resolve weights -> predict -> benchmark -> shared report.

Examples (run from your project root using .venv):
  python evaluate_yolo26.py --checkpoint runs/MY_RUN/weights/best.pt --imgsz 640
  python evaluate_yolo26.py --hf --hf-filename runs/MY_RUN/weights/best.pt --imgsz 640
See README.md for verified HF paths, pinned revisions and final-test rules.
"""
from pathlib import Path
import argparse
import gc
import os
import tempfile

import checkpoints
import evaluation_common as ev


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    ev.common_arguments(parser, "yolo26")
    parser.add_argument("--hf-filename", help="Exact .pt path inside the HF repo, preferably the validation-selected best.pt")
    parser.add_argument("--imgsz", type=int, default=640, help="Use 640 or 960 to match the training experiment")
    parser.add_argument("--batch", type=int, default=8)
    return parser.parse_args()


def main():
    args = arguments()
    ev.check_arguments(args)
    ev.require(args.imgsz > 0 and args.imgsz % 32 == 0 and args.batch >= 1, "Invalid image size/batch.")
    ev.require(args.hf or args.hf_filename is None, "Do not combine a local checkpoint and --hf-filename.")
    # Never install dependencies automatically. Missing packages must fail clearly.
    os.environ["YOLO_AUTOINSTALL"] = "false"
    import cv2
    import torch
    import ultralytics
    from ultralytics import YOLO
    ev.require(ultralytics.__version__ == ev.EXPECTED_ULTRALYTICS, "Use ultralytics==8.4.152.")
    ev.require(torch.cuda.is_available() and args.device < torch.cuda.device_count(), "Requested CUDA device unavailable.")
    torch.cuda.set_device(args.device)
    weights, provenance = checkpoints.resolve_yolo(
        local=args.checkpoint, repo=args.hf_repo, filename=args.hf_filename,
        revision=args.hf_revision, cache_dir=args.cache_dir)
    # quantize=16 and nms=None are the actual 8.4.152 API: do not replace them
    # with flags from an older Ultralytics version. This preserves FP16 + NMS.
    collection = {"imgsz": args.imgsz, "conf": ev.SCORE_FLOOR, "iou": 0.70,
                  "max_det": ev.MAX_DETECTIONS, "quantize": 16, "rect": False, "nms": None}
    settings = {"collection": collection, "prediction_batch": args.batch, "ultralytics": ultralytics.__version__,
                "opencv": cv2.__version__, "score_definition": "YOLO fracture confidence",
                "expected_precision": "FP16", "expected_head": "one-to-many with external NMS"}
    identity = ev.protocol_identity("yolo26", provenance, settings, [__file__, ev.__file__, checkpoints.__file__])
    frozen = ev.frozen_protocol(args, identity)  # Reject changed test settings BEFORE reading test images/labels.
    cohort, paths, labels = ev.evaluation_cohort(args)
    output = ev.create_output(args)
    detector = YOLO(str(weights))
    names = list(detector.names.values()) if isinstance(detector.names, dict) else list(detector.names)
    ev.require(len(names) == 1 and str(names[0]).lower() == "fracture", "Use the fine-tuned fracture model, not COCO weights.")
    metadata = cohort.set_index("filestem").to_dict("index")
    records, seen = [], set()
    # A temporary path list keeps image loading streamed; passing 2,895 decoded
    # images as a Python list would unnecessarily load the whole cohort into RAM.
    with tempfile.TemporaryDirectory(prefix="fracture-eval-") as temp:
        source = Path(temp) / "images.txt"
        source.write_text("\n".join(str(path) for path in paths.values()))
        predictions = detector.predict(source=str(source), stream=True, batch=args.batch,
                                       device=args.device, verbose=False, save=False, **collection)
        with (output / "predictions.jsonl").open("w", encoding="utf-8") as saved:
            for result in predictions:
                if not seen:
                    ev.require(bool(getattr(detector.predictor.model, "fp16", False)), "Expected actual FP16 predictions.")
                    ev.require(not bool(getattr(detector.predictor.model, "end2end", False)), "Expected one-to-many NMS head.")
                stem = Path(result.path).stem
                ev.require(stem in metadata and stem not in seen, f"Unexpected/duplicate image: {stem}")
                seen.add(stem)
                height, width = result.orig_shape
                ev.require((result.boxes.cls.cpu().numpy() == 0).all(), "Unexpected predicted class.")
                pred, scores = result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()
                gt = ev.ground_truth_xyxy(labels[stem], width, height)
                record = ev.make_record(stem, metadata[stem]["patient_id"], gt, pred, scores,
                                        "valid_nonempty" if len(pred) else "valid_empty")
                records.append(record)
                ev.save_prediction(saved, record, width, height, paths[stem])
                if len(records) % 25 == 0:
                    print(f"Predicted {len(records)}/{len(cohort)}", flush=True)
        ev.require(seen == set(cohort.filestem), "Incomplete image coverage.")
        del predictions, result
    # The same image order is required by both metric adapters and the original evaluator.
    records.sort(key=lambda r: r["stem"])
    if ev.finish_smoke(args, output, records, provenance, identity):
        return

    def timed_predict(image, cutoff):
        return detector.predict(source=image, **{**collection, "conf": cutoff}, batch=1,
                                device=args.device, verbose=False, save=False, stream=False)

    bench = ev.benchmark(timed_predict, list(paths.values()),
                         lambda p: cv2.imread(str(p), cv2.IMREAD_COLOR), args.device, args.benchmark_runs,
                         "FP16", "Preprocessing, inference, NMS and framework overhead", forward_phase=True)
    del detector
    gc.collect()
    torch.cuda.empty_cache()
    ev.finish(args, output, cohort, records, provenance, identity, frozen, bench)


if __name__ == "__main__":
    main()
