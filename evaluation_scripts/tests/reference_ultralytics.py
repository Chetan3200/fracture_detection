"""Minimal independently extracted test oracle, NOT an installed Ultralytics library.

Ultralytics 8.4.152 source: AGPL-3.0, https://github.com/ultralytics/ultralytics.
Only box_iou, smooth, compute_ap, ap_per_class and BaseValidator.match_predictions
are retained. Original operating functions come from the CURRENT attachment at
2026-09-18_YgeuHthvdMJU1XN4, never an older parent-session attachment.
Functions were selected by AST and copied verbatim (class indentation removed).
PROVENANCE pins both whole input files and each exact extracted function.
Tests always check vendored function hashes; they also check upstream bytes when
those local files exist. No imports of Ultralytics, GPU work, plotting or images.
"""
from pathlib import Path
from types import SimpleNamespace
import warnings
import numpy as np
import pandas as pd
import torch

# The only checks.check_version call retained by compute_ap is numpy >= 2.0.
def _check_version(current, required):
    assert required == ">=2.0", required
    return np.lib.NumpyVersion(current) >= np.lib.NumpyVersion("2.0.0")

checks = SimpleNamespace(check_version=_check_version)
OPERATING_IOU = 0.50
COLLECTION = dict(imgsz=640, conf=0.001, iou=0.70, max_det=300, quantize=16, rect=False, nms=None)

PROVENANCE = {'ultralytics_metrics': {'path': '/Users/chetan/.aside/u/0/sessions/2026-09-17_yPN42v0eEXQ7a1u8/tmp/ultralytics_hn_reference/wheel/ultralytics/utils/metrics.py', 'sha256': 'a2125a0e9269fbb2e186e0a972d35465a9a81049b0cd6b568448abb784587869', 'functions': {'box_iou': {'lineno': 82, 'end_lineno': 102, 'dedented_source_sha256': 'cd2d94236e8d2c72f1a3dad7411c9c39653840afa25c3b958c801d8e6e4bd196'}, 'smooth': {'lineno': 659, 'end_lineno': 664, 'dedented_source_sha256': 'e3d90d093f8605d327656d89e4981add63601291ad929293cd90343d93f5ae85'}, 'compute_ap': {'lineno': 760, 'end_lineno': 789, 'dedented_source_sha256': '9d00da3bc36a32df65b8d6ee6e19551fcf51c2e6b5385f1dba6a966006a8b864'}, 'ap_per_class': {'lineno': 792, 'end_lineno': 887, 'dedented_source_sha256': 'b9062b7183f9948ec980a2e3254bf80a4e1dec289836758d5f0597aad010fb1e'}}}, 'ultralytics_validator': {'path': '/Users/chetan/.aside/u/0/sessions/2026-09-17_yPN42v0eEXQ7a1u8/tmp/ultralytics_hn_reference/wheel/ultralytics/engine/validator.py', 'sha256': '3c34fed7b2974dd7054e0a800a3edc6c6f11b277ca4f5d80dadeaaf65903412e', 'functions': {'match_predictions': {'lineno': 308, 'end_lineno': 345, 'dedented_source_sha256': '8e6ec04f0681447d177a464d91e1285921ef9c0c2984d099755ac1c2079fb1ab'}}}, 'current_attachment': {'path': '/Users/chetan/.aside/u/0/sessions/2026-09-18_YgeuHthvdMJU1XN4/attachments/grazpedwri_evaluate_yolo26.py', 'sha256': 'e690b96d66d88f669aee017296c1a04806d540487a7b27847170d5cf08d8ef85', 'functions': {'operating_matches': {'lineno': 91, 'end_lineno': 107, 'dedented_source_sha256': '9b7b09d2559f4f289d12e0846f2b120306e81adb9341efceb22f329975c17ada'}, 'choose_threshold': {'lineno': 110, 'end_lineno': 175, 'dedented_source_sha256': 'a5046d632812adedf0838186f4e8139949cfdc9026931ee231c7e7cfde6e25f5'}, 'summarize': {'lineno': 178, 'end_lineno': 226, 'dedented_source_sha256': 'b8982da79d1767858f872103cb91f726bea1bea35b8e7eadfaf08be1047909f2'}}}}

# AST-extracted unchanged from ultralytics_metrics, lines 82-102.
# Exact dedented source SHA256: cd2d94236e8d2c72f1a3dad7411c9c39653840afa25c3b958c801d8e6e4bd196
def box_iou(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Calculate intersection-over-union (IoU) of boxes.

    Args:
        box1 (torch.Tensor): A tensor of shape (N, 4) representing N bounding boxes in (x1, y1, x2, y2) format.
        box2 (torch.Tensor): A tensor of shape (M, 4) representing M bounding boxes in (x1, y1, x2, y2) format.
        eps (float, optional): A small value to avoid division by zero.

    Returns:
        (torch.Tensor): An NxM tensor containing the pairwise IoU values for every element in box1 and box2.

    References:
        https://github.com/pytorch/vision/blob/main/torchvision/ops/boxes.py
    """
    # NOTE: Need .float() to get accurate iou values
    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1.float().unsqueeze(1).chunk(2, 2), box2.float().unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp_(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)


# AST-extracted unchanged from ultralytics_metrics, lines 659-664.
# Exact dedented source SHA256: e3d90d093f8605d327656d89e4981add63601291ad929293cd90343d93f5ae85
def smooth(y: np.ndarray, f: float = 0.05) -> np.ndarray:
    """Box filter of fraction f."""
    nf = round(len(y) * f * 2) // 2 + 1  # number of filter elements (must be odd)
    p = np.ones(nf // 2)  # ones padding
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)  # y padded
    return np.convolve(yp, np.ones(nf) / nf, mode="valid")  # y-smoothed


# AST-extracted unchanged from ultralytics_metrics, lines 760-789.
# Exact dedented source SHA256: 9d00da3bc36a32df65b8d6ee6e19551fcf51c2e6b5385f1dba6a966006a8b864
def compute_ap(recall: list[float], precision: list[float]) -> tuple[float, np.ndarray, np.ndarray]:
    """Compute the average precision (AP) given the recall and precision curves.

    Args:
        recall (list[float]): The recall curve.
        precision (list[float]): The precision curve.

    Returns:
        ap (float): Average precision.
        mpre (np.ndarray): Precision envelope curve.
        mrec (np.ndarray): Modified recall curve with sentinel values added at the beginning and end.
    """
    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.0], recall, [recall[-1] if len(recall) else 1.0], [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0], [0.0]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = "interp"  # methods: 'continuous', 'interp'
    if method == "interp":
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        func = np.trapezoid if checks.check_version(np.__version__, ">=2.0") else np.trapz  # np.trapz deprecated
        ap = func(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x-axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec


# AST-extracted unchanged from ultralytics_metrics, lines 792-887.
# Exact dedented source SHA256: b9062b7183f9948ec980a2e3254bf80a4e1dec289836758d5f0597aad010fb1e
def ap_per_class(
    tp: np.ndarray,
    conf: np.ndarray,
    pred_cls: np.ndarray,
    target_cls: np.ndarray,
    plot: bool = False,
    on_plot=None,
    save_dir: Path = Path(),
    names: dict[int, str] | None = None,
    eps: float = 1e-16,
    prefix: str = "",
) -> tuple:
    """Compute the average precision per class for object detection evaluation.

    Args:
        tp (np.ndarray): Binary array indicating whether the detection is correct (True) or not (False).
        conf (np.ndarray): Array of confidence scores of the detections.
        pred_cls (np.ndarray): Array of predicted classes of the detections.
        target_cls (np.ndarray): Array of true classes of the targets.
        plot (bool, optional): Whether to plot PR curves or not.
        on_plot (callable, optional): A callback to pass plots path and data when they are rendered.
        save_dir (Path, optional): Directory to save the PR curves.
        names (dict[int, str], optional): Dictionary of class names to plot PR curves.
        eps (float, optional): A small value to avoid division by zero.
        prefix (str, optional): A prefix string for saving the plot files.

    Returns:
        tp (np.ndarray): True positive counts at threshold given by max F1 metric for each class.
        fp (np.ndarray): False positive counts at threshold given by max F1 metric for each class.
        p (np.ndarray): Precision values at threshold given by max F1 metric for each class.
        r (np.ndarray): Recall values at threshold given by max F1 metric for each class.
        f1 (np.ndarray): F1-score values at threshold given by max F1 metric for each class.
        ap (np.ndarray): Average precision for each class at different IoU thresholds.
        unique_classes (np.ndarray): An array of unique classes that have data.
        p_curve (np.ndarray): Precision curves for each class.
        r_curve (np.ndarray): Recall curves for each class.
        f1_curve (np.ndarray): F1-score curves for each class.
        x (np.ndarray): X-axis values for the curves.
        prec_values (np.ndarray): Precision values at mAP@0.5 for each class.
    """
    names = names if names is not None else {}
    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    x, prec_values = np.linspace(0, 1, 1000), []

    # Average precision, precision and recall curves
    ap, p_curve, r_curve = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = nt[ci]  # number of labels
        n_p = i.sum()  # number of predictions
        if n_p == 0 or n_l == 0:
            prec_values.append(np.zeros_like(x))  # keep one row per class, aligned with `ap` and `names`
            continue

        # Accumulate FPs and TPs
        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)

        # Recall
        recall = tpc / (n_l + eps)  # recall curve
        r_curve[ci] = np.interp(-x, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases

        # Precision
        precision = tpc / (tpc + fpc)  # precision curve
        p_curve[ci] = np.interp(-x, -conf[i], precision[:, 0], left=1)  # p at pr_score

        # AP from recall-precision curve
        for j in range(tp.shape[1]):
            ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
            if j == 0:
                prec_values.append(np.interp(x, mrec, mpre))  # precision at mAP@0.5

    prec_values = np.array(prec_values) if prec_values else np.zeros((1, 1000))  # (nc, 1000)

    # Compute F1 (harmonic mean of precision and recall)
    f1_curve = 2 * p_curve * r_curve / (p_curve + r_curve + eps)
    names = {i: names[k] for i, k in enumerate(unique_classes) if k in names}  # dict: only classes that have data
    if plot:
        plot_pr_curve(x, prec_values, ap, save_dir / f"{prefix}PR_curve.png", names, on_plot=on_plot)
        plot_mc_curve(x, f1_curve, save_dir / f"{prefix}F1_curve.png", names, ylabel="F1", on_plot=on_plot)
        plot_mc_curve(x, p_curve, save_dir / f"{prefix}P_curve.png", names, ylabel="Precision", on_plot=on_plot)
        plot_mc_curve(x, r_curve, save_dir / f"{prefix}R_curve.png", names, ylabel="Recall", on_plot=on_plot)

    i = smooth(f1_curve.mean(0), 0.1).argmax()  # max F1 index
    p, r, f1 = p_curve[:, i], r_curve[:, i], f1_curve[:, i]  # max-F1 precision, recall, F1 values
    tp = (r * nt).round()  # true positives
    fp = (tp / (p + eps) - tp).round()  # false positives
    return tp, fp, p, r, f1, ap, unique_classes.astype(int), p_curve, r_curve, f1_curve, x, prec_values


# AST-extracted unchanged from ultralytics_validator, lines 308-345.
# Exact dedented source SHA256: 8e6ec04f0681447d177a464d91e1285921ef9c0c2984d099755ac1c2079fb1ab
def match_predictions(
    self, pred_classes: torch.Tensor, true_classes: torch.Tensor, iou: torch.Tensor, use_scipy: bool = False
) -> torch.Tensor:
    """Match predictions to ground truth objects using IoU.

    Args:
        pred_classes (torch.Tensor): Predicted class indices of shape (N,).
        true_classes (torch.Tensor): Target class indices of shape (M,).
        iou (torch.Tensor): An NxM tensor containing the pairwise IoU values for predictions and ground truth.
        use_scipy (bool, optional): Whether to use Hungarian one-to-one matching (more precise).

    Returns:
        (torch.Tensor): Correct tensor of shape (N, 10) for 10 IoU thresholds.
    """
    # Dx10 matrix, where D - detections, 10 - IoU thresholds
    correct = np.zeros((pred_classes.shape[0], self.iouv.shape[0])).astype(bool)
    # LxD matrix where L - labels (rows), D - detections (columns)
    correct_class = true_classes[:, None] == pred_classes
    iou = iou * correct_class  # zero out the wrong classes
    iou = iou.cpu().numpy()
    for i, threshold in enumerate(self.iouv.cpu().tolist()):
        if use_scipy:
            cost_matrix = iou * (iou >= threshold)
            if cost_matrix.any():
                labels_idx, detections_idx = linear_sum_assignment(-cost_matrix)  # negate to maximize IoU
                valid = cost_matrix[labels_idx, detections_idx] > 0
                if valid.any():
                    correct[detections_idx[valid], i] = True
        else:
            matches = np.nonzero(iou >= threshold)  # IoU > threshold and classes match
            matches = np.array(matches).T
            if matches.shape[0]:
                if matches.shape[0] > 1:
                    matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                    matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                    matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                correct[matches[:, 1].astype(int), i] = True
    return torch.from_numpy(correct)


# AST-extracted unchanged from current_attachment, lines 91-107.
# Exact dedented source SHA256: 9b7b09d2559f4f289d12e0846f2b120306e81adb9341efceb22f329975c17ada
def operating_matches(ious):
    """Predictions must be in descending-confidence order."""
    ng, npred = ious.shape
    correct = np.zeros(npred, dtype=bool)
    used = np.zeros(ng, dtype=bool)

    if ng:
        for j in range(npred):
            overlaps = ious[:, j].copy()
            overlaps[used] = -1
            k = int(overlaps.argmax())

            if overlaps[k] >= OPERATING_IOU:
                correct[j] = True
                used[k] = True

    return correct


# AST-extracted unchanged from current_attachment, lines 110-175.
# Exact dedented source SHA256: a5046d632812adedf0838186f4e8139949cfdc9026931ee231c7e7cfde6e25f5
def choose_threshold(records):
    scores = np.concatenate([r["scores"] for r in records])
    correct = np.concatenate([r["op_correct"] for r in records])
    n_gt = sum(r["n_gt"] for r in records)

    if not n_gt:
        raise ValueError(
            "Cannot choose an operating threshold without labeled fractures."
        )

    if not len(scores):
        warnings.warn(
            "No predictions: using confidence 1.0; "
            "this is not a useful operating point."
        )
        return 1.0, pd.DataFrame(
            columns=["confidence", "precision", "recall", "F1"]
        )

    order = np.argsort(-scores, kind="stable")
    scores, correct = scores[order], correct[order]

    # Include every prediction tied at a particular confidence.
    ends = np.flatnonzero(
        np.r_[scores[:-1] != scores[1:], True]
    )

    tp = correct.cumsum()[ends]
    n_pred = ends + 1
    precision = tp / n_pred
    recall = tp / n_gt
    f1 = 2 * tp / (n_pred + n_gt)

    # Scores must be STRICTLY GREATER than the cutoff.
    # Use the next excluded score as each candidate cutoff.
    cutoffs = np.array([
        float(scores[end + 1])
        if end + 1 < len(scores)
        else COLLECTION["conf"]
        for end in ends
    ])

    # Also allow rejecting all predictions.
    cutoffs = np.r_[1.0, cutoffs]
    precision = np.r_[0.0, precision]
    recall = np.r_[0.0, recall]
    f1 = np.r_[0.0, f1]

    # Candidates are ordered highest cutoff first.
    best = int(np.argmax(f1))

    if f1[best] == 0:
        warnings.warn(
            "No correct detections at IoU 0.50; selected F1 is zero."
        )

    table = pd.DataFrame(
        dict(
            confidence=cutoffs,
            precision=precision,
            recall=recall,
            F1=f1,
        )
    )

    return float(cutoffs[best]), table


# AST-extracted unchanged from current_attachment, lines 178-226.
# Exact dedented source SHA256: b8982da79d1767858f872103cb91f726bea1bea35b8e7eadfaf08be1047909f2
def summarize(records, threshold):
    n_images = len(records)
    n_gt = sum(r["n_gt"] for r in records)

    scores = np.concatenate([r["scores"] for r in records])
    ap_correct = np.concatenate(
        [r["ap_correct"] for r in records], axis=0
    )
    op_correct = np.concatenate(
        [r["op_correct"] for r in records]
    )

    keep = scores > threshold
    n_pred = int(keep.sum())
    tp = int(op_correct[keep].sum())
    fp = n_pred - tp

    recall = tp / n_gt if n_gt else np.nan
    precision = tp / n_pred if n_pred else 0.0
    f1 = (
        2 * tp / (n_pred + n_gt)
        if n_pred + n_gt
        else np.nan
    )

    if not n_gt:
        ap50, ap5095 = np.nan, np.nan
    elif not len(scores) or not ap_correct.any():
        ap50, ap5095 = 0.0, 0.0
    else:
        ap = ap_per_class(
            ap_correct,
            scores,
            np.zeros(len(scores), dtype=int),
            np.zeros(n_gt, dtype=int),
            plot=False,
        )[5]

        ap50 = float(ap[0, 0])
        ap5095 = float(ap[0].mean())

    return dict(
        AP50_95=ap5095,
        AP50=ap50,
        lesion_recall=recall,
        false_positives_per_image=fp / n_images,
        precision=precision,
        F1=f1,
    )
