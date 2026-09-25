"""Evaluate the fine-tuned MedGemma adapter using the SAME metric engine as YOLO.

  .venv-medgemma/bin/python evaluate_medgemma.py --hf
  .venv-medgemma/bin/python evaluate_medgemma.py --checkpoint runs/MY_RUN

Both evaluators use data/grazpedwri_yolo. MedGemma images are decoded exactly
as in its training-data preparation: OpenCV IMREAD_COLOR, then BGR to RGB.
Never directly convert a raw 16-bit PIL image to RGB: that clips the X-ray.
"""
from pathlib import Path
import argparse
import gc

import checkpoints
import evaluation_common as ev
import medgemma_model


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    ev.common_arguments(parser, "medgemma")
    parser.add_argument("--hf-run", help="Run directory name in HF; auto-select only when one completed full run exists")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Defaults to the saved training value (768); frozen between validation/test")
    return parser.parse_args()


def read_rgb(path):
    import cv2
    from PIL import Image
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    ev.require(bgr is not None, f"Cannot decode image: {path}")
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def main():
    args = arguments()
    ev.check_arguments(args)
    ev.require(args.hf or args.hf_run is None, "Do not combine a local checkpoint and --hf-run.")
    import cv2
    import torch
    ev.require(torch.cuda.is_available() and args.device < torch.cuda.device_count(), "Requested CUDA device unavailable.")
    run, adapter, provenance = checkpoints.resolve_medgemma(
        local=args.checkpoint, repo=args.hf_repo, run=args.hf_run,
        revision=args.hf_revision, cache_dir=args.cache_dir)
    predictor = medgemma_model.MedGemmaPredictor(run, adapter, args.device, args.max_new_tokens)
    settings = {**predictor.settings, "opencv": cv2.__version__,
                "source_image_decode": "OpenCV IMREAD_COLOR uint8 BGR -> RGB; native dimensions, no other change"}
    # Device numbers are locations, not prediction settings. GPU name is recorded
    # in the benchmark; moving the identical model to device 1 must remain possible.
    settings.pop("device_index", None)
    identity = ev.protocol_identity("medgemma", provenance, settings,
                                   [__file__, ev.__file__, checkpoints.__file__, medgemma_model.__file__])
    frozen = ev.frozen_protocol(args, identity)
    cohort, paths, labels = ev.evaluation_cohort(args)
    output = ev.create_output(args)
    records = []
    with (output / "predictions.jsonl").open("w", encoding="utf-8") as saved:
        for index, row in enumerate(cohort.itertuples(index=False), start=1):
            image = read_rgb(paths[row.filestem])
            try:
                width, height = image.size
                # No labels, demographics or reference answers enter generation.
                prediction = predictor.predict(image)
            finally:
                image.close()
            gt = ev.ground_truth_xyxy(labels[row.filestem], width, height)
            record = ev.make_record(row.filestem, row.patient_id, gt, prediction["pred_xyxy"],
                                    prediction["scores"], prediction["status"])
            records.append(record)
            details = {key: value for key, value in prediction.items() if key not in ("pred_xyxy", "scores", "status")}
            details["emitted_box_count_before_collection"] = len(prediction["scores"])
            details["emitted_box_scores"] = prediction["scores"]
            ev.save_prediction(saved, record, width, height, paths[row.filestem], details)
            print(f"{index}/{len(cohort)} {row.filestem}: {prediction['status']}, "
                  f"{len(record['scores'])} collected boxes, {prediction['generated_tokens']} tokens", flush=True)

    if ev.finish_smoke(args, output, records, provenance, identity):
        return

    def timed_predict(image, cutoff):
        result = predictor.predict(image)
        # Post-generation score filtering only: unlike YOLO, it cannot reduce
        # MedGemma's decoding time or produce additional candidate boxes.
        keep = sorted((i for i, score in enumerate(result["scores"]) if score > cutoff),
                      key=lambda i: -result["scores"][i])[:ev.MAX_DETECTIONS]
        result["pred_xyxy"] = [result["pred_xyxy"][i] for i in keep]
        result["scores"] = [result["scores"][i] for i in keep]
        return result

    bench = ev.benchmark(timed_predict, list(paths.values()), read_rgb, args.device, args.benchmark_runs,
                         predictor.settings["precision"],
                         "Preprocessing, greedy autoregressive generation, box scoring/parsing, filtering and framework overhead")
    del predictor
    gc.collect()
    torch.cuda.empty_cache()
    ev.finish(args, output, cohort, records, provenance, identity, frozen, bench)


if __name__ == "__main__":
    main()
