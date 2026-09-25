#!/usr/bin/env python3
"""Draw saved YOLO/MedGemma predictions; CPU only, no inference or installation.

Input: one completed evaluator folder (predictions.jsonl + evaluation.json)
       and the original prepared data/grazpedwri_yolo images.
Output: two-panel PNGs and index.csv in a NEW overlays_TIMESTAMP folder.
Left: ground truth only. Right: ground truth plus all SAVED prediction boxes.
The JSONL already contains original-image pixel XYXY coordinates. Do NOT
normalize them again or interpret them as the raw model's YXYX coordinates.

Example:
  .venv/bin/python overlay_predictions.py --evaluation runs/evaluation_medgemma_val_... --limit 40
  # --limit 0 draws every evaluated image; --image-id STEM selects one (repeatable).
Dependencies: existing OpenCV and Pillow only. Never loads Torch or model weights.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import csv
import hashlib
import json
import math
import random
import re
import sys
import textwrap

GT_COLOR = (65, 230, 105)
KEPT_COLOR = (255, 90, 100)
BELOW_COLOR = (255, 190, 65)
BACKGROUND = (18, 22, 29)
TEXT_COLOR = (235, 238, 245)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def validate_record(record):
    stem = record.get('filestem')
    require(isinstance(stem, str) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', stem),
            'Missing or unsafe filestem in predictions.jsonl.')
    for key in ('width', 'height'):
        require(type(record.get(key)) is int and record[key] > 0, f'{stem}: invalid {key}.')
    for key in ('gt_xyxy', 'pred_xyxy'):
        boxes = record.get(key)
        require(isinstance(boxes, list), f'{stem}: missing {key}.')
        for box in boxes:
            require(isinstance(box, list) and len(box) == 4 and all(number(v) for v in box),
                    f'{stem}: invalid pixel XYXY box in {key}.')
            require(box[0] <= box[2] and box[1] <= box[3], f'{stem}: reversed box in {key}.')
    scores = record.get('scores')
    require(isinstance(scores, list) and len(scores) == len(record['pred_xyxy']) and
            all(number(s) and 0 <= s <= 1 for s in scores), f'{stem}: invalid scores.')
    require(isinstance(record.get('status'), str), f'{stem}: missing generation status.')
    require(isinstance(record.get('image_sha256'), str) and
            re.fullmatch(r'[0-9a-f]{64}', record['image_sha256']), f'{stem}: missing image checksum.')


def load_evaluation(folder):
    folder = Path(folder).resolve()
    metadata = json.loads((folder / 'evaluation.json').read_text())
    require(metadata.get('status') == 'complete', 'Use a completed evaluation, not a four-image smoke run.')
    require(metadata.get('split') in ('val', 'test'), 'Unknown evaluation split.')
    protocol = metadata['protocol']
    cutoff = protocol['confidence_cutoff']
    require(number(cutoff) and 0 <= cutoff <= 1, 'Invalid frozen score cutoff.')
    require(protocol['identity'].get('confidence_comparison') == 'score > cutoff',
            'Unsupported cutoff comparison; expected the shared evaluator format.')
    source = folder / 'predictions.jsonl'
    require(sha256(source) == metadata.get('predictions_sha256'),
            'predictions.jsonl does not match evaluation.json. Use files from the SAME evaluation.')
    records, seen = [], set()
    with source.open(encoding='utf-8') as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                validate_record(record)
            except (ValueError, TypeError, AttributeError) as error:
                raise ValueError(f'Invalid prediction record on line {line_number}: {error}') from error
            require(record['filestem'] not in seen, f'Duplicate image: {record["filestem"]}')
            seen.add(record['filestem'])
            records.append(record)
    require(len(records) == metadata['counts']['images'] and len(records) > 0,
            'Prediction count does not match completed evaluation.')
    return metadata, records, float(cutoff)


def select_records(records, limit, seed, image_ids):
    require(limit >= 0, '--limit must be nonnegative; 0 means all images.')
    indexed = {r['filestem']: r for r in records}
    if image_ids:
        missing = set(image_ids) - set(indexed)
        require(not missing, f'Image IDs are not in this evaluation: {sorted(missing)}')
        return [indexed[stem] for stem in sorted(set(image_ids))]
    ordered = sorted(records, key=lambda r: r['filestem'])
    selected = random.Random(seed).sample(ordered, min(limit, len(ordered))) if limit else ordered
    return sorted(selected, key=lambda r: r['filestem'])


def image_paths(dataset, split, selected):
    directory = Path(dataset).resolve() / 'images' / split
    require(directory.is_dir(), f'Image directory not found: {directory}. Set --dataset to the prepared YOLO dataset.')
    wanted = {r['filestem'] for r in selected}
    indexed = {}
    for path in directory.iterdir():
        if path.stem in wanted and path.is_file():
            indexed.setdefault(path.stem, []).append(path)
    result = {}
    for record in selected:
        stem = record['filestem']
        candidates = indexed.get(stem, [])
        require(len(candidates) == 1, f'{stem}: expected one original image, found {len(candidates)} in {directory}.')
        path = candidates[0]
        require(sha256(path) == record['image_sha256'], f'{stem}: image checksum differs from the evaluated image.')
        result[stem] = path
    return result


def read_rgb(path):
    # Match evaluation/training decoding. Direct PIL conversion of raw 16-bit
    # X-rays may clip intensities and make an otherwise correct overlay misleading.
    import cv2
    from PIL import Image
    cv2.setNumThreads(1)
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    require(bgr is not None, f'Cannot decode image: {path}')
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def iou(a, b):
    intersection = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return intersection / (area_a + area_b - intersection + 1e-7)


def best_iou(box, ground_truth):
    # Diagnostic nearest-GT overlap only, NOT the evaluator's one-to-one matching.
    return max((iou(box, gt) for gt in ground_truth), default=0.0)


def scaled_box(box, original_size, display_size):
    sx, sy = display_size[0] / original_size[0], display_size[1] / original_size[1]
    return (box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy)


def dashed_box(draw, box, color, width=2, dash=9):
    x0, y0, x1, y1 = box
    for a, b in (((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)),
                 ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))):
        dx, dy = b[0] - a[0], b[1] - a[1]
        length = math.hypot(dx, dy)
        if length == 0:
            continue
        for start in range(0, math.ceil(length), 2 * dash):
            end = min(length, start + dash)
            draw.line((a[0] + dx * start / length, a[1] + dy * start / length,
                       a[0] + dx * end / length, a[1] + dy * end / length), fill=color, width=width)


def font(size=16):
    from PIL import ImageFont
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow 10.0 fallback; no system font installation required.
        return ImageFont.load_default()


def label(draw, xy, text, color, canvas_size, face):
    bounds = draw.textbbox((0, 0), text, font=face)
    tw, th = bounds[2] - bounds[0], bounds[3] - bounds[1]
    x = max(0, min(xy[0], canvas_size[0] - tw - 8))
    y = max(0, min(xy[1] - th - 10, canvas_size[1] - th - 8))
    draw.rectangle((x, y, x + tw + 8, y + th + 8), fill=(0, 0, 0))
    draw.text((x + 4 - bounds[0], y + 4 - bounds[1]), text, fill=color, font=face)


def render_overlay(image, record, cutoff, max_side=1200):
    from PIL import Image, ImageDraw
    original = (record['width'], record['height'])
    require(image.size == original, f'{record["filestem"]}: decoded dimensions {image.size} differ from saved {original}.')
    require(max_side >= 0, '--max-side must be nonnegative; 0 means native resolution.')
    scale = min(1.0, max_side / max(original)) if max_side else 1.0
    size = (max(1, round(original[0] * scale)), max(1, round(original[1] * scale)))
    base = image.convert('RGB')
    if base.size != size:
        resized = base.resize(size, Image.Resampling.LANCZOS)
        base.close()
        base = resized
    left, right = base.copy(), base.copy()
    base.close()
    dl, dr = ImageDraw.Draw(left), ImageDraw.Draw(right)
    face = font(16)
    # Low-score boxes first, so they cannot hide the retained predictions.
    for j, (box, score) in enumerate(zip(record['pred_xyxy'], record['scores']), 1):
        if score <= cutoff:
            shown = scaled_box(box, original, size)
            dashed_box(dr, shown, BELOW_COLOR)
            label(dr, shown[:2], f'P{j} s={score:.6f} maxIoU={best_iou(box, record["gt_xyxy"]):.2f}', BELOW_COLOR, size, face)
    for j, box in enumerate(record['gt_xyxy'], 1):
        shown = scaled_box(box, original, size)
        for draw in (dl, dr):
            draw.rectangle(shown, outline=GT_COLOR, width=3)
            label(draw, shown[:2], f'G{j}', GT_COLOR, size, face)
    for j, (box, score) in enumerate(zip(record['pred_xyxy'], record['scores']), 1):
        if score > cutoff:
            shown = scaled_box(box, original, size)
            dr.rectangle(shown, outline=KEPT_COLOR, width=2)
            label(dr, shown[:2], f'P{j} s={score:.6f} maxIoU={best_iou(box, record["gt_xyxy"]):.2f}', KEPT_COLOR, size, face)
    kept = sum(score > cutoff for score in record['scores'])
    out_of_bounds = sum(box[0] < 0 or box[1] < 0 or box[2] > original[0] or box[3] > original[1]
                        for box in record['gt_xyxy'] + record['pred_xyxy'])
    header = [(record['filestem'], TEXT_COLOR),
              (f'status={record["status"]} | GT={len(record["gt_xyxy"])} | kept={kept} | below/equal cutoff={len(record["scores"]) - kept}', TEXT_COLOR),
              (f'Frozen rule: score > {cutoff:.8f} | original {original[0]}x{original[1]} | out-of-frame boxes={out_of_bounds}', TEXT_COLOR),
              ('GREEN = ground truth', GT_COLOR),
              ('SOLID RED = retained prediction; DASHED AMBER = saved prediction below/equal cutoff', BELOW_COLOR),
              ('maxIoU = best overlap with any GT, NOT one-to-one TP matching. Scores are not clinical probabilities.', TEXT_COLOR)]
    # Wrap long IDs/legends even for unusually small images.
    panel_width = max(size[0], 480)
    total_width, margin, gap, line_height = 2 * panel_width + 36, 12, 12, 23
    wrapped = [(line, color) for text, color in header
               for line in textwrap.wrap(text, width=max(35, total_width // 9))]
    top = 16 + line_height * (len(wrapped) + 1)
    canvas = Image.new('RGB', (total_width, top + size[1] + 45), BACKGROUND)
    dc = ImageDraw.Draw(canvas)
    y = 10
    for text, color in wrapped:
        dc.text((margin, y), text, fill=color, font=face)
        y += line_height
    dc.text((margin, top - line_height), 'GROUND TRUTH ONLY', fill=GT_COLOR, font=face)
    dc.text((margin + panel_width + gap, top - line_height), 'GROUND TRUTH + SAVED PREDICTIONS', fill=TEXT_COLOR, font=face)
    canvas.paste(left, (margin, top))
    canvas.paste(right, (margin + panel_width + gap, top))
    dc.text((margin, top + size[1] + 10), 'Display-only resize. Saved boxes only: collection floor/cap already applied. No new inference.', fill=TEXT_COLOR, font=face)
    left.close()
    right.close()
    return canvas


def arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--evaluation', type=Path, required=True, help='Completed evaluator output directory')
    parser.add_argument('--dataset', type=Path, default=Path.cwd() / 'data/grazpedwri_yolo')
    parser.add_argument('--limit', type=int, default=40, help='Reproducible random image sample; 0 means all (default: 40)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--image-id', action='append', default=[], help='Exact filestem; repeat for multiple images; overrides sampling')
    parser.add_argument('--max-side', type=int, default=1200, help='Display-only maximum image side; 0 means native resolution')
    parser.add_argument('--output', type=Path, help='NEW directory; defaults to evaluation/overlays_TIMESTAMP')
    return parser.parse_args()


def main():
    args = arguments()
    require(args.max_side >= 0, '--max-side must be nonnegative.')
    # Missing dependencies fail clearly. Never install or alter the environment.
    try:
        import cv2
        from PIL import Image
    except ImportError as error:
        raise RuntimeError('Use the existing evaluation .venv, which already contains OpenCV and Pillow. No automatic installation.') from error
    metadata, records, cutoff = load_evaluation(args.evaluation)
    selected = select_records(records, args.limit, args.seed, args.image_id)
    paths = image_paths(args.dataset, metadata['split'], selected)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%fZ')
    output = (args.output or args.evaluation / f'overlays_{stamp}').resolve()
    output.mkdir(parents=True, exist_ok=False)
    columns = ['file', 'filestem', 'patient_id', 'status', 'gt_boxes', 'saved_predictions',
               'kept_predictions', 'below_or_equal_cutoff', 'confidence_cutoff', 'image_path']
    with (output / 'index.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, columns)
        writer.writeheader()
        for index, record in enumerate(selected, 1):
            image = read_rgb(paths[record['filestem']])
            try:
                overlay = render_overlay(image, record, cutoff, args.max_side)
                try:
                    filename = record['filestem'] + '.png'
                    overlay.save(output / filename)
                finally:
                    overlay.close()
            finally:
                image.close()
            kept = sum(score > cutoff for score in record['scores'])
            writer.writerow(dict(file=filename, filestem=record['filestem'], patient_id=record.get('patient_id'),
                                 status=record['status'], gt_boxes=len(record['gt_xyxy']),
                                 saved_predictions=len(record['scores']), kept_predictions=kept,
                                 below_or_equal_cutoff=len(record['scores']) - kept,
                                 confidence_cutoff=cutoff, image_path=str(paths[record['filestem']])))
            stream.flush()
            print(f'{index}/{len(selected)} {filename}: {record["status"]}; GT={len(record["gt_xyxy"])} kept={kept}', flush=True)
    print(f'\nSaved {len(selected)} overlay PNGs + index.csv to: {output}')
    print('Evaluation, predictions, images and model checkpoints were not modified. No inference was run.')


if __name__ == '__main__':
    try:
        main()
    except (ValueError, RuntimeError, FileNotFoundError, FileExistsError, KeyError) as error:
        sys.exit(f'ERROR: {error}')
