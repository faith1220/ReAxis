# Copyright (c) 2026 ReAxis contributors
# SPDX-License-Identifier: GPL-2.0-only
"""Independent CPU evaluation for image, pixel, and sample anomaly scores.

Scores are returned on the 0--1 scale. Larger prediction scores must mean more
anomalous. Ground truths must be binary (0 or 1); scores need not be probabilities.
Non-finite selected inputs raise ValueError rather than being silently discarded.

I-* measures images; P-* measures all selected pixels pooled together. S-* first
groups images by sample_ids, taking the maximum label and maximum score within
each sample. AUROC and Overkill require both classes. AP and F1max require at
least one positive, but remain defined for an all-positive population. Undefined
metrics and metrics for an empty class are NaN.

P-AUPRO is the normalized area of macro connected-region overlap against
background false-positive rate up to 0.3. Regions are 8-connected components
within each 2-D image and receive equal weight regardless of their areas. The
curve uses the empty prediction and at most aupro_num_thresholds distinct score
thresholds, sampled at evenly spaced ranks among distinct descending scores;
the largest and smallest scores are included. A threshold includes all scores
greater than or equal to it, including ties. Linear interpolation supplies the
FPR=0.3 endpoint. Vertical curve segments are retained during integration.
Missing foreground regions or background pixels makes AUPRO undefined.

Overkill@k is the smallest normal false-positive rate among score thresholds
whose anomaly false-negative rate is at most k/100. This uses deterministic
thresholds (no randomized splitting of tied scores), for k in {2, 5, 10}.

This evaluator is an independent implementation using scikit-learn's BSD
metrics and NumPy/SciPy connected components. Its threshold sampling, region
connectivity, aggregation, and undefined-value rules do not guarantee numerical
agreement with any previous evaluator. AUPRO needs O(number of pixels) working
memory, not a pixels-by-thresholds prediction tensor. The device argument is
accepted for call compatibility; tensors are detached and evaluated on CPU.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math

import numpy as np
from scipy import ndimage
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
import torch


_BASIC_METRICS = ("AUROC", "AP", "F1max")
_OVERKILL_METRICS = ("Overkill@2", "Overkill@5", "Overkill@10")
_DEFAULT_METRICS = (
    "I-AUROC", "I-AP", "I-F1max",
    "P-AUROC", "P-AP", "P-F1max", "P-AUPRO",
)
_SUPPORTED_METRICS = {
    f"{level}-{name}"
    for level in ("I", "P", "S")
    for name in _BASIC_METRICS
} | {
    f"{level}-{name}"
    for level in ("I", "S")
    for name in _OVERKILL_METRICS
} | {"P-AUPRO"}


def _numeric_array(value, name: str) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        if value.is_complex():
            raise ValueError(f"{name} must contain real numeric values")
        # Converting before numpy also supports CUDA and bfloat16 tensors.
        value = value.detach().to(device="cpu", dtype=torch.float64).numpy()
    array = np.asarray(value)
    if array.dtype.kind not in "biuf":
        raise ValueError(f"{name} must contain real numeric values")
    return array


def _check_values(array: np.ndarray, name: str, *, binary: bool) -> None:
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinity in the selected class")
    if binary and not np.logical_or(array == 0, array == 1).all():
        raise ValueError(f"{name} must contain only binary values 0 and 1")


def _vector(results: Mapping, name: str, selected: np.ndarray,
            *, binary: bool) -> np.ndarray:
    array = _numeric_array(results[name], name)
    if array.ndim != 1 or array.shape[0] != selected.size:
        raise ValueError(f"{name} must have shape [N], matching cls_names")
    array = array[selected]
    _check_values(array, name, binary=binary)
    return array.astype(np.uint8 if binary else np.float64, copy=False)


def _masks(results: Mapping, name: str, selected: np.ndarray,
           *, binary: bool) -> np.ndarray:
    array = _numeric_array(results[name], name)
    if array.ndim == 4 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim != 3 or array.shape[0] != selected.size:
        raise ValueError(f"{name} must have shape [N,H,W] or [N,1,H,W]")
    if array.shape[1] == 0 or array.shape[2] == 0:
        raise ValueError(f"{name} must have nonempty spatial dimensions")
    array = array[selected]
    _check_values(array, name, binary=binary)
    return array.astype(np.uint8 if binary else np.float64, copy=False)


def _classification_metrics(labels: np.ndarray, scores: np.ndarray,
                            requested: Sequence[str]) -> dict[str, float]:
    values = dict.fromkeys(requested, math.nan)
    positives = int(np.count_nonzero(labels))
    negatives = labels.size - positives
    if not labels.size:
        return values
    if "AUROC" in values and positives and negatives:
        values["AUROC"] = float(roc_auc_score(labels, scores))
    if "AP" in values and positives:
        values["AP"] = float(average_precision_score(labels, scores))
    if "F1max" in values and positives:
        precision, recall, _ = precision_recall_curve(labels, scores)
        denominator = precision + recall
        f1 = np.divide(2 * precision * recall, denominator,
                       out=np.zeros_like(denominator), where=denominator > 0)
        values["F1max"] = float(f1.max())
    overkill = [name for name in requested if name in _OVERKILL_METRICS]
    if overkill and positives and negatives:
        fpr, tpr, _ = roc_curve(labels, scores, drop_intermediate=False)
        for name in overkill:
            allowed_misses = int(name.split("@", 1)[1]) / 100.0
            eligible = tpr >= 1.0 - allowed_misses - 1e-12
            values[name] = float(fpr[eligible].min())
    return values


def _sample_max(labels: np.ndarray, scores: np.ndarray, sample_ids) -> tuple:
    grouped = {}
    for label, score, sample_id in zip(labels, scores, sample_ids):
        if sample_id is None or (
            isinstance(sample_id, (float, np.floating))
            and not math.isfinite(float(sample_id))
        ):
            raise ValueError("sample_ids must be nonmissing, finite, hashable IDs")
        try:
            old = grouped.get(sample_id)
            if old is None:
                grouped[sample_id] = (int(label), float(score))
            else:
                grouped[sample_id] = (max(old[0], int(label)),
                                      max(old[1], float(score)))
        except TypeError as error:
            raise ValueError("sample_ids must contain hashable IDs") from error
    pairs = list(grouped.values())
    return (np.asarray([pair[0] for pair in pairs], dtype=np.uint8),
            np.asarray([pair[1] for pair in pairs], dtype=np.float64))


def _aupro(gt_masks: np.ndarray, scores: np.ndarray,
           num_thresholds: int) -> float:
    foreground = gt_masks.astype(bool, copy=False)
    background_count = foreground.size - int(np.count_nonzero(foreground))
    if not background_count or not foreground.any():
        return math.nan

    # Each pixel in region r contributes 1 / |r| before averaging regions.
    weights = np.zeros(foreground.shape, dtype=np.float64)
    region_count = 0
    connectivity = ndimage.generate_binary_structure(2, 2)
    for image_index, mask in enumerate(foreground):
        components, count = ndimage.label(mask, structure=connectivity)
        if count:
            areas = np.bincount(components.ravel())
            reciprocal_area = np.zeros(areas.size, dtype=np.float64)
            reciprocal_area[1:] = 1.0 / areas[1:]
            weights[image_index] = reciprocal_area[components]
            region_count += count

    # Sort once, then query cumulative counts at tie-group boundaries. This
    # avoids materializing one binary prediction map for every threshold.
    order = np.argsort(scores.ravel(), kind="stable")[::-1]
    ordered_scores = scores.ravel()[order]
    group_ends = np.concatenate((
        np.flatnonzero(ordered_scores[:-1] != ordered_scores[1:]),
        np.asarray([ordered_scores.size - 1]),
    ))
    if group_ends.size > num_thresholds:
        ranks = np.linspace(0, group_ends.size - 1, num_thresholds,
                            dtype=np.int64)
        group_ends = group_ends[ranks]
    background_prefix = np.cumsum(~foreground.ravel()[order], dtype=np.int64)
    overlap_prefix = np.cumsum(weights.ravel()[order], dtype=np.float64)
    fpr = np.concatenate(([0.0], background_prefix[group_ends] / background_count))
    pro = np.concatenate(([0.0], overlap_prefix[group_ends] / region_count))
    pro = np.clip(pro, 0.0, 1.0)  # Protect against cumulative floating-point drift.

    limit = 0.3
    stop = int(np.searchsorted(fpr, limit, side="right"))
    x = fpr[:stop]
    y = pro[:stop]
    if x[-1] < limit:
        fraction = (limit - x[-1]) / (fpr[stop] - x[-1])
        boundary_overlap = y[-1] + fraction * (pro[stop] - y[-1])
        x = np.concatenate((x, [limit]))
        y = np.concatenate((y, [boundary_overlap]))
    return float(np.clip(auc(x, y) / limit, 0.0, 1.0))


class Evaluator:
    """Evaluate one class from a results dictionary.

    With metrics=None the seven I/P metrics are requested. sample_level=True
    additionally requests the three S metrics in that default case. An explicit
    metrics sequence is used as given; requesting S-* enables sample aggregation
    regardless of sample_level. Duplicate metric names are evaluated only once.
    Missing result fields raise KeyError; invalid shapes/values raise ValueError.
    Only fields required by the requested metrics are accessed. logger, if given,
    must provide an info(message) method.
    """

    def __init__(self, device, metrics=None, sample_level=False,
                 aupro_num_thresholds=None):
        if aupro_num_thresholds is None:
            aupro_num_thresholds = 200
        if (isinstance(aupro_num_thresholds, (bool, np.bool_))
                or not isinstance(aupro_num_thresholds, (int, np.integer))
                or aupro_num_thresholds < 2):
            raise ValueError("aupro_num_thresholds must be an integer >= 2")
        if metrics is None:
            metrics = list(_DEFAULT_METRICS)
            if sample_level:
                metrics.extend(f"S-{name}" for name in _BASIC_METRICS)
        if isinstance(metrics, (str, bytes)) or not isinstance(metrics, Sequence):
            raise ValueError("metrics must be a sequence of metric names")
        if any(not isinstance(name, str) or name not in _SUPPORTED_METRICS
               for name in metrics):
            raise ValueError(f"Unsupported metric; choose from {sorted(_SUPPORTED_METRICS)}")
        self.device = device
        self.metrics = tuple(dict.fromkeys(metrics))
        self.sample_level = bool(sample_level)
        self.aupro_num_thresholds = int(aupro_num_thresholds)

    def run(self, results, cls_name, logger=None) -> dict[str, float]:
        if not isinstance(results, Mapping):
            raise ValueError("results must be a mapping")
        classes = np.asarray(results["cls_names"])
        if classes.ndim != 1 or any(not isinstance(name, str) for name in classes):
            raise ValueError("cls_names must be a one-dimensional string array")
        if not isinstance(cls_name, str):
            raise ValueError("cls_name must be a string")
        selected = classes == cls_name
        values = dict.fromkeys(self.metrics, math.nan)
        if selected.any() and self.metrics:
            groups = {
                level: [name.split("-", 1)[1] for name in self.metrics
                        if name.startswith(f"{level}-")]
                for level in ("I", "P", "S")
            }
            if groups["I"] or groups["S"]:
                labels = _vector(results, "gt_anomalys", selected, binary=True)
                scores = _vector(results, "pr_anomalys", selected, binary=False)
                if groups["I"]:
                    values.update({f"I-{name}": value for name, value in
                                   _classification_metrics(labels, scores, groups["I"]).items()})
                if groups["S"]:
                    ids = results["sample_ids"]
                    try:
                        valid_ids_shape = len(ids) == selected.size
                    except TypeError:
                        valid_ids_shape = False
                    if isinstance(ids, np.ndarray) and ids.ndim != 1:
                        valid_ids_shape = False
                    if isinstance(ids, (str, bytes)) or not valid_ids_shape:
                        raise ValueError("sample_ids must have N entries matching cls_names")
                    chosen_ids = [sample_id for sample_id, keep in zip(ids, selected) if keep]
                    sample_labels, sample_scores = _sample_max(labels, scores, chosen_ids)
                    values.update({f"S-{name}": value for name, value in
                                   _classification_metrics(sample_labels, sample_scores,
                                                           groups["S"]).items()})
            if groups["P"]:
                gt_masks = _masks(results, "gt_masks", selected, binary=True)
                pr_masks = _masks(results, "pr_masks", selected, binary=False)
                if gt_masks.shape != pr_masks.shape:
                    raise ValueError("gt_masks and pr_masks must have matching spatial shapes")
                pixel_metrics = [name for name in groups["P"] if name != "AUPRO"]
                values.update({f"P-{name}": value for name, value in
                               _classification_metrics(gt_masks.ravel(), pr_masks.ravel(),
                                                       pixel_metrics).items()})
                if "AUPRO" in groups["P"]:
                    values["P-AUPRO"] = _aupro(gt_masks, pr_masks, self.aupro_num_thresholds)
        if logger is not None:
            logger.info(f"{cls_name}: " + ", ".join(
                f"{name}={value:.6g}" for name, value in values.items()))
        return values
