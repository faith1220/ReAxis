# Copyright (c) 2026 ReAxis contributors
# SPDX-License-Identifier: GPL-2.0-only
"""Synthetic definition-based checks for the independent public evaluator."""

import math

import numpy as np
import pytest
import torch

from tools.effecient_metric import Evaluator


def image_results(labels=(0, 0, 1, 1), scores=(0.1, 0.4, 0.35, 0.8)):
    return {
        "cls_names": np.asarray(["widget"] * len(labels)),
        "gt_anomalys": torch.tensor(labels),
        "pr_anomalys": torch.tensor(scores, dtype=torch.float64),
    }


def pixel_results(mask, scores, channel=False):
    gt = torch.as_tensor(mask)
    pr = torch.as_tensor(scores, dtype=torch.float64)
    if gt.ndim == 2:
        gt, pr = gt.unsqueeze(0), pr.unsqueeze(0)
    if channel:
        gt, pr = gt.unsqueeze(1), pr.unsqueeze(1)
    return {"cls_names": np.asarray(["widget"] * len(gt)),
            "gt_masks": gt, "pr_masks": pr}


@pytest.mark.parametrize("prefix", ["I", "P"])
def test_basic_metrics_from_hand_calculated_ranking(prefix):
    results = image_results() if prefix == "I" else pixel_results(
        [[0, 0, 1, 1]], [[0.1, 0.4, 0.35, 0.8]])
    result = Evaluator("cpu", [f"{prefix}-{name}" for name in
                               ("AUROC", "AP", "F1max")]).run(results, "widget")
    assert result[f"{prefix}-AUROC"] == pytest.approx(3 / 4)
    assert result[f"{prefix}-AP"] == pytest.approx((1 + 2 / 3) / 2)
    assert result[f"{prefix}-F1max"] == pytest.approx(4 / 5)
    assert all(isinstance(value, float) for value in result.values())


def test_tied_scores_include_whole_group():
    result = Evaluator("cpu", ["I-AUROC", "I-AP", "I-F1max", "I-Overkill@2"]).run(
        image_results(scores=(0.5, 0.5, 0.5, 0.5)), "widget")
    assert result == pytest.approx({"I-AUROC": 0.5, "I-AP": 0.5,
                                    "I-F1max": 2 / 3, "I-Overkill@2": 1.0})


@pytest.mark.parametrize("channel", [False, True])
def test_aupro_perfect_and_reversed(channel):
    mask = [[1, 0, 0], [0, 0, 1]]
    perfect = pixel_results(mask, mask, channel=channel)
    reverse = pixel_results(mask, 1 - np.asarray(mask), channel=channel)
    evaluator = Evaluator("cpu", ["P-AUPRO"])
    assert evaluator.run(perfect, "widget")["P-AUPRO"] == pytest.approx(1.0)
    assert evaluator.run(reverse, "widget")["P-AUPRO"] == pytest.approx(0.0)


def test_aupro_ties_are_linearly_interpolated_at_cutoff():
    result = Evaluator("cpu", ["P-AUPRO"]).run(
        pixel_results([[1, 0], [0, 0]], np.full((2, 2), 0.5)), "widget")
    # A single tied score gives the segment (FPR, PRO)=(0,0)--(1,1).
    assert result["P-AUPRO"] == pytest.approx(0.3 / 2)


def test_aupro_weights_regions_equally_not_pixels():
    mask = [[1, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 1, 1]]
    scores = [[1.0, 0.5, 0.5, 0.5, 0.5], [0.5] * 5,
              [0.5, 0.5, 0.5, 0.0, 0.0]]
    result = Evaluator("cpu", ["P-AUPRO"]).run(pixel_results(mask, scores), "widget")
    # The one-pixel region is found at FPR=0; the two-pixel region only at FPR=1.
    assert result["P-AUPRO"] == pytest.approx(0.5)


def test_aupro_retains_vertical_segments_and_partial_boundary():
    mask = [[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    scores = [[3, 1, 4, 2, 0, 0, 0, 0, 0, 0, 0, 0]]
    # Curve: (0,0), (.1,0), (.1,.5), (.2,.5), (.2,1), (1,1).
    # Up to .3: area .1*.5 + .1*1 = .15, normalized to .5.
    result = Evaluator("cpu", ["P-AUPRO"]).run(pixel_results(mask, scores), "widget")
    assert result["P-AUPRO"] == pytest.approx(0.5)


def test_aupro_uses_eight_connected_regions():
    mask = [[1, 0, 0], [0, 1, 1], [0, 1, 1]]
    scores = [[1, 0.5, 0.5], [0.5, 0, 0], [0.5, 0, 0]]
    result = Evaluator("cpu", ["P-AUPRO"]).run(pixel_results(mask, scores), "widget")
    # Diagonal contact joins the singleton and four-pixel block into one region.
    assert result["P-AUPRO"] == pytest.approx(1 / 5)


@pytest.mark.parametrize("value", [0, 1])
def test_aupro_is_undefined_without_both_foreground_and_background(value):
    result = Evaluator("cpu", ["P-AUPRO"]).run(
        pixel_results(np.full((2, 2), value), np.zeros((2, 2))), "widget")
    assert math.isnan(result["P-AUPRO"])


@pytest.mark.parametrize("label", [0, 1])
def test_single_class_metric_definitions(label):
    result = Evaluator("cpu", ["I-AUROC", "I-AP", "I-F1max", "I-Overkill@5"]).run(
        image_results(labels=(label, label), scores=(0.1, 0.9)), "widget")
    assert math.isnan(result["I-AUROC"])
    assert math.isnan(result["I-Overkill@5"])
    for name in ("I-AP", "I-F1max"):
        if label:
            assert result[name] == pytest.approx(1.0)
        else:
            assert math.isnan(result[name])


def test_empty_class_returns_nan_without_requiring_unused_fields():
    result = Evaluator("cpu").run({"cls_names": np.array(["other"])}, "widget")
    assert len(result) == 7
    assert all(math.isnan(value) for value in result.values())


def test_empty_dataset_returns_nan():
    result = Evaluator("cpu", ["I-AUROC"]).run({"cls_names": np.array([], dtype=str)}, "widget")
    assert math.isnan(result["I-AUROC"])


def test_class_filtering_happens_before_value_validation():
    results = image_results(labels=(0, 1, 1), scores=(0.1, 0.9, float("nan")))
    results["cls_names"] = np.asarray(["widget", "widget", "other"])
    assert Evaluator("cpu", ["I-AUROC"]).run(results, "widget")["I-AUROC"] == 1.0


def test_sample_aggregation_uses_maximum_label_and_score():
    results = image_results(labels=(0, 1, 0, 0), scores=(0.9, 0.1, 0.5, 0.4))
    results["sample_ids"] = ["a", "a", "b", "b"]
    result = Evaluator("cpu", ["I-AUROC", "S-AUROC", "S-AP", "S-F1max", "S-Overkill@2"]).run(
        results, "widget")
    assert result["I-AUROC"] == 0.0
    assert all(result[name] == 1.0 for name in ("S-AUROC", "S-AP", "S-F1max"))
    assert result["S-Overkill@2"] == 0.0


def test_sample_level_flag_extends_defaults_but_not_explicit_metrics():
    assert len(Evaluator("cpu", sample_level=True).metrics) == 10
    assert Evaluator("cpu", ["I-AP"], sample_level=True).metrics == ("I-AP",)


def test_default_metrics_run_together_on_zero_to_one_scale():
    results = image_results(labels=(0, 1), scores=(0.1, 0.9))
    results.update(pixel_results([[[0, 0]], [[1, 0]]],
                                 [[[0.1, 0.2]], [[0.9, 0.3]]], channel=True))
    results["sample_ids"] = ["normal", "anomalous"]
    result = Evaluator("cpu", sample_level=True).run(results, "widget")
    assert len(result) == 10
    assert all(value == pytest.approx(1.0) for value in result.values())


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cpu_tensor_scores_are_detached_and_converted(dtype):
    results = image_results()
    results["pr_anomalys"] = results["pr_anomalys"].to(dtype).requires_grad_()
    value = Evaluator("cpu", ["I-AUROC"]).run(results, "widget")["I-AUROC"]
    assert value == pytest.approx(0.75)


def test_overkill_limits_anomaly_misses_at_observed_thresholds():
    # A normal item outranks exactly the lowest-scoring 5% of anomalies.
    labels = [1] * 20 + [0, 0]
    scores = list(range(1, 21)) + [1.5, 0]
    result = Evaluator("cpu", ["I-Overkill@2", "I-Overkill@5", "I-Overkill@10"]).run(
        image_results(labels, scores), "widget")
    assert result == {"I-Overkill@2": 0.5, "I-Overkill@5": 0.0, "I-Overkill@10": 0.0}


@pytest.mark.parametrize("count", [0, 1, -2, 2.0, True, "200"])
def test_invalid_threshold_count_is_rejected(count):
    with pytest.raises(ValueError, match="integer >= 2"):
        Evaluator("cpu", aupro_num_thresholds=count)


def test_threshold_sampling_remains_bounded_and_deterministic():
    mask = np.asarray([[1, 0, 0, 1, 0], [0, 0, 0, 0, 0]])
    scores = np.arange(10, dtype=float).reshape(2, 5)
    evaluator = Evaluator("cpu", ["P-AUPRO"], aupro_num_thresholds=np.int64(2))
    results = pixel_results(mask, scores)
    first = evaluator.run(results, "widget")["P-AUPRO"]
    assert 0 <= first <= 1
    assert first == evaluator.run(results, "widget")["P-AUPRO"]


@pytest.mark.parametrize("metric", ["I-Accuracy", "P-Overkill@5", "I-Overkill@3", 3])
def test_unknown_metric_is_rejected(metric):
    with pytest.raises(ValueError, match="Unsupported metric"):
        Evaluator("cpu", [metric])


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_scores_are_rejected(bad):
    with pytest.raises(ValueError, match="NaN or infinity"):
        Evaluator("cpu", ["I-AUROC"]).run(image_results(scores=(0.1, bad, 0.5, 0.9)), "widget")
    with pytest.raises(ValueError, match="NaN or infinity"):
        Evaluator("cpu", ["P-AUPRO"]).run(pixel_results([[0, 1]], [[0.1, bad]]), "widget")


def test_invalid_labels_and_mask_shapes_are_rejected():
    with pytest.raises(ValueError, match="binary"):
        Evaluator("cpu", ["I-AUROC"]).run(image_results(labels=(0, 2), scores=(0, 1)), "widget")
    results = pixel_results([[0, 1]], [[0.1, 0.9]])
    results["pr_masks"] = torch.zeros((1, 2, 1, 2))
    with pytest.raises(ValueError, match="shape"):
        Evaluator("cpu", ["P-AUPRO"]).run(results, "widget")
    results["pr_masks"] = torch.zeros((1, 2, 2))
    with pytest.raises(ValueError, match="matching spatial"):
        Evaluator("cpu", ["P-AUPRO"]).run(results, "widget")


def test_vector_length_and_missing_fields_are_rejected():
    results = image_results()
    results["pr_anomalys"] = torch.zeros(3)
    with pytest.raises(ValueError, match="shape"):
        Evaluator("cpu", ["I-AP"]).run(results, "widget")
    del results["pr_anomalys"]
    with pytest.raises(KeyError, match="pr_anomalys"):
        Evaluator("cpu", ["I-AP"]).run(results, "widget")


@pytest.mark.parametrize("ids", [None, 7, ["a"], [None] * 4,
                                 [float("nan")] * 4, [[1]] * 4, np.zeros((4, 1))])
def test_invalid_sample_ids_are_rejected(ids):
    results = image_results()
    results["sample_ids"] = ids
    with pytest.raises(ValueError, match="sample_ids"):
        Evaluator("cpu", ["S-AUROC"]).run(results, "widget")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is optional")
def test_cuda_inputs_are_detached_and_evaluated():
    results = image_results()
    results["gt_anomalys"] = results["gt_anomalys"].cuda()
    results["pr_anomalys"] = results["pr_anomalys"].cuda().requires_grad_()
    assert Evaluator("cuda", ["I-AUROC"]).run(results, "widget")["I-AUROC"] == pytest.approx(0.75)
