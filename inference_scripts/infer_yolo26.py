"""YOLO26 fracture inference on an image or folder, without labels or metrics.

  ./.venv/bin/python fracture_evaluation/infer_yolo26.py --checkpoint best.pt \
      --source radiographs/ --protocol runs/YOUR_VAL/evaluation.json
  ./.venv/bin/python fracture_evaluation/infer_yolo26.py --hf \
      --hf-filename runs/YOUR_RUN/epoch_100/best.pt --source radiograph.png \
      --imgsz 960 --conf 0.37

Use --protocol for the frozen validation cutoff; --conf is explicitly exploratory.
"""
import argparse
import gc
import os
from pathlib import Path
import sys

# Must be set before Ultralytics is imported. Never install missing dependencies.
os.environ['YOLO_AUTOINSTALL'] = 'false'

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from evaluation_scripts import checkpoints
import inference_common as ic

EXPECTED_ULTRALYTICS = '8.4.152'


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    ic.add_arguments(parser, 'yolo26')
    parser.add_argument('--hf-filename', help='Exact checkpoint path inside the HF model repository')
    parser.add_argument('--imgsz', type=int, default=None, help='Defaults to the validation protocol value, or 640 with explicit --conf')
    parser.add_argument('--batch', type=int, default=None, help='Defaults to the validation protocol value, or 8 with explicit --conf')
    return parser.parse_args()


def main():
    args = arguments()
    ic.validate_args(args, 'yolo26')
    ic.require(not args.hf or args.hf_filename, '--hf requires --hf-filename.')
    report = ic.read_protocol(args.protocol, 'yolo26')
    cutoff = ic.resolve_cutoff(args, report)
    imgsz = ic.option_from_protocol(args.imgsz, report, ('collection', 'imgsz'), 640)
    batch = ic.option_from_protocol(args.batch, report, ('prediction_batch',), 8)
    ic.require(type(imgsz) is int and imgsz > 0 and imgsz % 32 == 0 and type(batch) is int and batch >= 1,
               'Use a positive image size divisible by 32 and a positive batch size.')
    files, source_root, output = ic.plan_inputs(args.source, args.recursive, args.output, 'yolo26')

    import cv2
    import torch
    import ultralytics
    from ultralytics import YOLO
    ic.require(ultralytics.__version__ == EXPECTED_ULTRALYTICS, f'Use existing ultralytics=={EXPECTED_ULTRALYTICS}; no packages were installed.')
    ic.require(torch.cuda.is_available() and args.device < torch.cuda.device_count(), 'Requested CUDA device unavailable.')
    torch.cuda.set_device(args.device)
    weights, provenance = checkpoints.resolve_yolo(
        local=args.checkpoint, repo=args.hf_repo, filename=args.hf_filename,
        revision=args.hf_revision, cache_dir=args.cache_dir)
    collection = {'imgsz': imgsz, 'conf': ic.SCORE_FLOOR, 'iou': 0.70,
                  'max_det': ic.MAX_DETECTIONS, 'quantize': 16, 'rect': False, 'nms': None}
    settings = {'collection': collection, 'prediction_batch': batch, 'ultralytics': ultralytics.__version__,
                'opencv': cv2.__version__, 'score_definition': 'YOLO fracture confidence',
                'expected_precision': 'FP16', 'expected_head': 'one-to-many with external NMS'}
    ic.verify_protocol(report, 'yolo26', provenance, settings, {'torch': str(torch.__version__)},
                       {'checkpoints.py': checkpoints.__file__})
    detector = YOLO(str(weights))
    names = list(detector.names.values()) if isinstance(detector.names, dict) else list(detector.names)
    ic.require(len(names) == 1 and str(names[0]).lower() == 'fracture',
               'Use the fine-tuned single-class fracture checkpoint, not COCO weights.')
    print(f'Checkpoint: {weights}\nImage size: {imgsz}; requested batch: {batch}; cutoff: {cutoff}\nOutput: {output}', flush=True)
    try:
        with ic.InferenceWriter(output, 'yolo26', source_root, files, cutoff, provenance, settings,
                                args.protocol, not args.no_annotate,
                                code_paths=[__file__, ic.__file__, checkpoints.__file__]) as writer:
            for start in range(0, len(files), batch):
                paths = files[start:start + batch]
                # Decode only one batch in memory. Passing BGR arrays avoids PIL
                # conversion/clipping of 16-bit images in Ultralytics list-of-paths input.
                images = [ic.read_bgr(p) for p in paths]
                results = detector.predict(source=images, stream=False, batch=batch, device=args.device,
                                           verbose=False, save=False, **collection)
                ic.require(len(results) == len(paths), 'YOLO returned an incomplete batch.')
                ic.require(bool(getattr(detector.predictor.model, 'fp16', False)), 'Expected actual FP16 inference.')
                ic.require(not bool(getattr(detector.predictor.model, 'end2end', False)), 'Expected one-to-many NMS head.')
                for index, (image_path, bgr, result) in enumerate(zip(paths, images, results), start=start + 1):
                    ic.require(tuple(result.orig_shape) == tuple(bgr.shape[:2]), 'Result/image dimensions do not match.')
                    classes = result.boxes.cls.detach().cpu().tolist()
                    ic.require(all(c == 0 for c in classes), 'Unexpected predicted class.')
                    boxes = result.boxes.xyxy.detach().cpu().tolist()
                    scores = result.boxes.conf.detach().cpu().tolist()
                    ic.require(len(classes) == len(boxes) == len(scores), 'YOLO class/box/score count mismatch.')
                    prediction = {'pred_xyxy': boxes, 'scores': scores,
                                  'status': 'valid_nonempty' if boxes else 'valid_empty'}
                    record = writer.add(image_path, bgr, prediction)
                    print(f"{index}/{len(files)} {image_path.name}: {record['status']}, {len(record['detections'])} boxes", flush=True)
                del results, images
        print(f'Inference complete: {output}', flush=True)
    finally:
        del detector
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
