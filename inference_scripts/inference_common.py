"""Inference-only I/O, score filtering and validation-protocol checks.

No datasets, labels, metric calculation, dependency installation or model loading.
Existing evaluator files are deliberately not imported or modified.
"""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import math
import struct

SCORE_FLOOR = 0.001
MAX_DETECTIONS = 300
EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}
SCORE_TYPES = {'yolo26': 'yolo_detector_score', 'medgemma': 'coordinate_token_likelihood_proxy'}
IDENTITY_KEYS = ('weights_sha256', 'config_sha256', 'base_model', 'base_revision', 'prompt_sha256', 'processor_sha256')
VALID_STATUSES = {'valid_empty', 'valid_nonempty'}
FAILURE_STATUSES = {'format_failure', 'schema_failure'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def add_arguments(parser, model):
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--checkpoint', type=Path, help='Local YOLO .pt or complete MedGemma run/adapter/checkpoint directory/recovery ZIP')
    source.add_argument('--hf', action='store_true', help='Resolve/download from the existing private Hugging Face checkpoint repo')
    parser.add_argument('--hf-repo', help='Optional model-repository override')
    parser.add_argument('--hf-revision', help='Commit/branch/tag, resolved to an immutable commit by the checkpoint loader')
    parser.add_argument('--cache-dir', type=Path, default=Path.home() / '.cache/fracture_evaluation')
    parser.add_argument('--source', type=Path, required=True, help='One image or a directory of images; not a DICOM file or a glob')
    parser.add_argument('--recursive', action='store_true', help='Include nested directories')
    parser.add_argument('--output', type=Path, help='NEW output directory outside the source directory; default runs/inference_MODEL_TIMESTAMP')
    parser.add_argument('--device', type=int, default=0, help='Logical CUDA index after CUDA_VISIBLE_DEVICES')
    cutoff = parser.add_mutually_exclusive_group(required=True)
    cutoff.add_argument('--protocol', type=Path, help='This checkpoint\'s COMPLETE validation evaluation.json; uses its fixed cutoff/settings')
    cutoff.add_argument('--conf', type=float, help='Explicit exploratory score cutoff, not a clinical probability; mutually exclusive with --protocol')
    parser.add_argument('--no-annotate', action='store_true', help='Only write JSONL and run metadata, not annotated PNGs')
    parser.set_defaults(model_name=model)


def validate_args(args, model):
    require(model in SCORE_TYPES and args.device >= 0, 'Invalid model or CUDA device.')
    require(args.hf or (args.hf_repo is None and args.hf_revision is None and
                       getattr(args, 'hf_run', None) is None and getattr(args, 'hf_filename', None) is None),
            'Local checkpoints cannot be combined with Hugging Face options.')
    if args.conf is not None:
        require(math.isfinite(args.conf) and SCORE_FLOOR <= args.conf <= 1,
                f'--conf must be finite and between {SCORE_FLOOR} and 1. Scores use a strict > comparison.')
    require((args.protocol is None) != (args.conf is None), 'Choose exactly one of --protocol or --conf.')


def read_protocol(path, model):
    if path is None:
        return None
    report = json.loads(Path(path).read_text(encoding='utf-8'))
    require(isinstance(report, dict) and report.get('status') == 'complete' and report.get('split') == 'val',
            'Use a complete validation evaluation.json, not a test, smoke or legacy report.')
    protocol = report.get('protocol', {})
    require(isinstance(protocol, dict), 'Missing/invalid protocol object.')
    identity = protocol.get('identity', {})
    require(isinstance(identity, dict), 'Missing/invalid protocol identity.')
    require(protocol.get('selected_on') == 'val' and identity.get('model') == model,
            'Validation protocol belongs to a different model or was not selected on validation.')
    cutoff = protocol.get('confidence_cutoff')
    require(type(cutoff) in (int, float) and math.isfinite(cutoff) and SCORE_FLOOR <= cutoff <= 1,
            'Invalid validation cutoff.')
    require(identity.get('score_floor') == SCORE_FLOOR and identity.get('max_detections') == MAX_DETECTIONS and
            identity.get('confidence_comparison') == 'score > cutoff', 'Unsupported validation collection/filtering rules.')
    return report


def resolve_cutoff(args, report):
    return float(report['protocol']['confidence_cutoff'] if report is not None else args.conf)


def option_from_protocol(explicit, report, keys, default):
    if report is None:
        return default if explicit is None else explicit
    value = report['protocol']['identity']['inference']
    for key in keys:
        require(isinstance(value, dict) and key in value, f'Protocol lacks inference setting: {".".join(keys)}')
        value = value[key]
    require(explicit is None or explicit == value,
            f'Explicit {".".join(keys)} differs from validation. Omit it to use the saved value.')
    return value


def verify_protocol(report, model, provenance, settings, software, helper_paths):
    """Check prediction-relevant identity, not metric code or input-cohort identity.

    These are NEW inference entry points, so their own hashes intentionally are
    not compared with the old evaluator entry-point hashes. Their hashes are
    recorded in inference.json. No claim of an identical dataset/evaluation run.
    """
    if report is None:
        return
    identity = report['protocol']['identity']
    current = {key: provenance[key] for key in IDENTITY_KEYS if key in provenance}
    require(identity.get('model') == model and identity.get('checkpoint') == current,
            'Checkpoint identity/hash differs from the validation protocol. Use the exact validated checkpoint bytes.')
    require(identity.get('inference') == settings, 'Inference settings or model/processor package versions differ from validation.')
    for name, version in software.items():
        require(identity.get('versions', {}).get(name) == version, f'{name} version differs from validation.')
    for name, filename in helper_paths.items():
        require(identity.get('source_sha256', {}).get(name) == sha256(filename),
                f'{name} differs from the helper used for validation.')


def plan_inputs(source, recursive, output, model):
    source = Path(source).expanduser().resolve()
    require(source.exists(), f'Input does not exist: {source}')
    if source.is_file():
        require(source.suffix.lower() in EXTENSIONS, 'Unsupported image format. Export a single-frame image first; DICOM/windowing is not implemented.')
        files, root = [source], source.parent
    else:
        require(source.is_dir(), f'Not a file or directory: {source}')
        root = source
        entries = source.rglob('*') if recursive else source.iterdir()
        files = []
        for item in entries:
            if item.is_file() and item.suffix.lower() in EXTENSIONS:
                resolved = item.resolve()
                require(resolved.is_relative_to(root), f'Image symlink escapes source directory: {item}')
                files.append(resolved)
        files = sorted(set(files), key=lambda p: p.relative_to(root).as_posix())
    require(files, f'No supported images in {source}; use --recursive for subdirectories.')
    if output is None:
        stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%fZ')
        output = Path.cwd() / 'runs' / f'inference_{model}_{stamp}'
    output = Path(output).expanduser().resolve()
    require(not output.exists(), f'Output already exists; choose a new directory: {output}')
    if source.is_dir():
        require(not output.is_relative_to(source), 'Put outputs outside the source directory to avoid reprocessing previous overlays.')
    require(all(not p.is_relative_to(output) for p in files), 'Output cannot contain the input images.')
    return files, root, output


def read_bgr(path):
    import cv2
    if Path(path).suffix.lower() in {'.tif', '.tiff'}:
        require(hasattr(cv2, 'imcount') and cv2.imcount(str(path)) == 1,
                f'Only single-frame TIFF files are supported: {path}')
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    require(image is not None and image.ndim == 3 and image.shape[2] == 3,
            f'Cannot decode an RGB-compatible image: {path}')
    return image


def float32(value):
    require(type(value) in (float, int) and math.isfinite(value), 'Non-numeric/nonfinite model output.')
    try:
        result = struct.unpack('<f', struct.pack('<f', value))[0]
    except (OverflowError, struct.error) as exc:
        raise ValueError('Model output exceeds float32 range.') from exc
    require(math.isfinite(result), 'Model output exceeds float32 range.')
    return result


def select_detections(boxes, scores, cutoff, width, height):
    """Match evaluation collection: float32, stable score sort, >.001, cap300, >cutoff."""
    require(len(boxes) == len(scores), 'Box/score count mismatch.')
    # NumPy 2.x evaluation compares float32 arrays with weak Python scalars in
    # float32. Round the floor/cutoff as well, including explicit decimal cutoffs.
    floor32, cutoff32 = float32(SCORE_FLOOR), float32(cutoff)
    candidates = []
    for box, score in zip(boxes, scores):
        require(len(box) == 4, 'Expected four xyxy coordinates.')
        b, s = [float32(v) for v in box], float32(score)
        require(0 <= s <= 1, 'Score outside [0,1].')
        require(0 <= b[0] <= b[2] <= width and 0 <= b[1] <= b[3] <= height,
                'Box is reversed or outside the decoded image.')
        if s > floor32:
            detection = {'label': 'fracture', 'xyxy': b, 'score': s}
            # Native YOLO can clip low-score border boxes to zero area. The
            # evaluator retains them; do not abort or silently change selection.
            if b[0] == b[2] or b[1] == b[3]:
                detection['degenerate'] = True
            candidates.append(detection)
    candidates.sort(key=lambda x: -x['score'])
    at_limit = len(candidates) >= MAX_DETECTIONS
    candidates = candidates[:MAX_DETECTIONS]
    return [x for x in candidates if x['score'] > cutoff32], len(candidates), at_limit


def make_record(image_path, relative, prediction, cutoff, width, height, model):
    raw_status = prediction.get('status')
    require(raw_status in VALID_STATUSES | FAILURE_STATUSES, f'Unknown model output status: {raw_status}')
    boxes, scores = prediction['pred_xyxy'], prediction['scores']
    if raw_status in FAILURE_STATUSES or raw_status == 'valid_empty':
        require(not boxes and not scores, 'Empty/invalid model output unexpectedly includes boxes.')
    if raw_status == 'valid_nonempty':
        require(len(boxes) > 0, 'Nonempty status without boxes.')
    detections, collected, at_limit = select_detections(boxes, scores, cutoff, width, height)
    status = ('invalid_model_output' if raw_status in FAILURE_STATUSES else
              'boxes_above_cutoff' if detections else 'no_boxes_above_cutoff')
    record = {'image': relative.as_posix(), 'input_path': str(image_path), 'image_sha256': sha256(image_path),
              'width': width, 'height': height, 'coordinate_format': 'xyxy_pixels_in_decoded_original_image',
              'status': status, 'model_output_status': raw_status, 'score_type': SCORE_TYPES[model],
              'confidence_cutoff': cutoff, 'confidence_comparison': 'score > cutoff',
              'raw_box_count': len(boxes), 'collected_box_count': collected, 'at_collection_limit': at_limit,
              'detections': detections}
    for key in ('generated_text', 'generated_tokens', 'truncated', 'error'):
        if key in prediction:
            record[key] = prediction[key]
    return record


def annotate(bgr, record, destination):
    import cv2
    canvas = bgr.copy()
    height, width = canvas.shape[:2]
    thickness = max(1, round(min(width, height) / 450))
    scale = max(0.35, min(0.8, min(width, height) / 1000))
    for detection in record['detections']:
        x1, y1, x2, y2 = detection['xyxy']
        p1 = (max(0, min(width-1, round(x1))), max(0, min(height-1, round(y1))))
        p2 = (max(0, min(width-1, round(x2))), max(0, min(height-1, round(y2))))
        cv2.rectangle(canvas, p1, p2, (0, 0, 255), thickness)
        kind = 'proxy' if record['score_type'] == SCORE_TYPES['medgemma'] else 'score'
        label = 'degenerate box' if detection.get('degenerate') else 'fracture'
        text = f"{label} {kind}={detection['score']:.3f}"
        cv2.putText(canvas, text, (p1[0], max(15, p1[1]-6)), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (0, 0, 255), thickness, cv2.LINE_AA)
    message = ('INVALID MODEL OUTPUT - not a negative' if record['status'] == 'invalid_model_output' else
               'No boxes above cutoff - not a diagnosis' if not record['detections'] else
               f"{len(record['detections'])} boxes above cutoff - research only")
    cv2.putText(canvas, message, (5, min(height-1, 22)), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 180, 255), thickness, cv2.LINE_AA)
    destination.parent.mkdir(parents=True, exist_ok=True)
    require(cv2.imwrite(str(destination), canvas), f'Cannot write annotated image: {destination}')


class InferenceWriter:
    def __init__(self, output, model, source_root, files, cutoff, provenance, settings,
                 protocol_path=None, annotate_images=True, code_paths=()):
        self.output, self.root, self.model = Path(output), Path(source_root), model
        self.files, self.expected, self.seen = list(files), set(files), set()
        self.cutoff, self.annotate_images = cutoff, annotate_images
        self.counts, self.handle = Counter(), None
        self.metadata = {'status': 'running', 'created_utc': utc_now(), 'model': model,
                         'input_root': str(self.root), 'input_count': len(files),
                         'confidence_cutoff': cutoff, 'confidence_comparison': 'score > cutoff',
                         'cutoff_source': 'validation_protocol' if protocol_path else 'explicit_unvalidated_override',
                         'validation_protocol': str(Path(protocol_path).resolve()) if protocol_path else None,
                         'validation_protocol_sha256': sha256(protocol_path) if protocol_path else None,
                         'checkpoint_source': provenance, 'inference': settings,
                         'score_type': SCORE_TYPES[model], 'score_is_clinical_probability': False,
                         'collection': {'score_floor': SCORE_FLOOR, 'max_detections': MAX_DETECTIONS,
                                        'score_dtype': 'float32', 'comparison_dtype': 'float32'},
                         'source_sha256': {Path(p).name: sha256(p) for p in code_paths},
                         'annotation_enabled': annotate_images, 'ground_truth_used': False,
                         'metrics_computed': False,
                         'note': 'Research inference only. No boxes above cutoff does not establish absence of fracture. '
                                 'Protocol checks prediction settings/helpers, not an identical evaluation dataset or metric run.'}

    def __enter__(self):
        self.output.mkdir(parents=True, exist_ok=False)
        write_json(self.output / 'inference.json', self.metadata)
        self.handle = (self.output / 'predictions.jsonl').open('x', encoding='utf-8')
        return self

    def add(self, image_path, bgr, prediction):
        image_path = Path(image_path)
        require(image_path in self.expected and image_path not in self.seen, 'Unexpected or duplicate image result.')
        height, width = bgr.shape[:2]
        require(width > 0 and height > 0, 'Empty decoded image.')
        relative = image_path.relative_to(self.root)
        record = make_record(image_path, relative, prediction, self.cutoff, width, height, self.model)
        if self.annotate_images:
            # Preserve the input extension: foo.jpg and foo.png must not collide.
            destination = self.output / 'annotated' / relative.parent / (relative.name + '.png')
            annotate(bgr, record, destination)
            record['annotated_image'] = destination.relative_to(self.output).as_posix()
        self.handle.write(json.dumps(record, allow_nan=False) + '\n')
        self.handle.flush()
        self.seen.add(image_path)
        self.counts[record['status']] += 1
        return record

    def __exit__(self, exc_type, exc, traceback):
        if self.handle:
            self.handle.close()
        complete = exc_type is None and self.seen == self.expected
        self.metadata.update(status='complete' if complete else 'failed', finished_utc=utc_now(),
                             processed_images=len(self.seen), status_counts=dict(self.counts),
                             predictions_sha256=sha256(self.output / 'predictions.jsonl'))
        if exc_type:
            self.metadata['error_type'] = exc_type.__name__
            self.metadata['error'] = str(exc)
        elif not complete:
            self.metadata['error'] = 'Incomplete image coverage.'
        write_json(self.output / 'inference.json', self.metadata)
        if exc_type is None:
            require(complete, 'Incomplete image coverage; partial outputs marked failed.')
        return False
