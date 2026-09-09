"""Numerical contracts for the public focal-loss implementation."""

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest
import torch


SPEC = importlib.util.spec_from_file_location(
    "public_reaxis_loss", Path(__file__).resolve().parents[1] / "adaptcliplib" / "loss.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
FocalLoss = MODULE.FocalLoss


def focal_value(probability, weight=1.0):
    return -weight * (1.0 - probability) ** 2 * math.log(probability)


def focal_derivative(probability, weight=1.0):
    # Analytic derivative with respect to the effective probability, gamma=2.
    return weight * (
        2.0 * (1.0 - probability) * math.log(probability)
        - (1.0 - probability) ** 2 / probability
    )


def test_unsmoothed_binary_value_and_gradient():
    probabilities = torch.tensor([[0.8, 0.2], [0.3, 0.7]], dtype=torch.float64, requires_grad=True)
    loss = FocalLoss(smooth=0)(probabilities, torch.tensor([0, 1]))
    expected = (focal_value(0.8) + focal_value(0.7)) / 2
    assert loss.item() == pytest.approx(expected, rel=1e-12)
    loss.backward()
    expected_gradient = torch.tensor(
        [[focal_derivative(0.8) / 2, 0.0], [0.0, focal_derivative(0.7) / 2]],
        dtype=torch.float64,
    )
    torch.testing.assert_close(probabilities.grad, expected_gradient, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("smoothing", [1e-5, 0.1, 0.8])
def test_smoothed_binary_value_and_gradient(smoothing):
    probabilities = torch.tensor([[0.8, 0.2]], dtype=torch.float64, requires_grad=True)
    loss = FocalLoss(smooth=smoothing)(probabilities, torch.tensor([[0]]))
    high = float(np.float32(1 - smoothing))
    low = float(np.float32(min(smoothing, 1 - smoothing)))
    q = high * 0.8 + low * 0.2 + smoothing
    assert loss.item() == pytest.approx(focal_value(q), rel=1e-12, abs=1e-15)
    loss.backward()
    expected_gradient = torch.tensor(
        [[high * focal_derivative(q), low * focal_derivative(q)]], dtype=torch.float64
    )
    torch.testing.assert_close(probabilities.grad, expected_gradient, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("weights", [[1, 3], np.array([1, 3])])
def test_class_weights_are_normalized(weights):
    probabilities = torch.tensor([[0.8, 0.2], [0.3, 0.7]], dtype=torch.float64)
    loss = FocalLoss(alpha=weights, smooth=0, size_average=False)(probabilities, torch.tensor([0, 1]))
    torch.testing.assert_close(
        loss,
        torch.tensor([focal_value(0.8, 0.25), focal_value(0.7, 0.75)], dtype=torch.float64),
        rtol=1e-12,
        atol=1e-12,
    )


def test_float_alpha_respects_balance_index():
    probabilities = torch.tensor([[0.8, 0.2], [0.3, 0.7]], dtype=torch.float64)
    loss = FocalLoss(alpha=0.25, balance_index=1, smooth=0, size_average=False)(
        probabilities, torch.tensor([0, 1])
    )
    torch.testing.assert_close(
        loss,
        torch.tensor([focal_value(0.8, 0.75), focal_value(0.7, 0.25)], dtype=torch.float64),
        rtol=1e-12,
        atol=1e-12,
    )


def test_spatial_layout_and_unreduced_order():
    probabilities = torch.tensor(
        [[[[0.8, 0.4], [0.3, 0.6]], [[0.2, 0.6], [0.7, 0.4]]]], dtype=torch.float64
    ).transpose(-1, -2)
    labels = torch.tensor([[[0, 1], [1, 0]]]).transpose(-1, -2)
    actual = FocalLoss(smooth=0, size_average=False)(probabilities, labels)
    expected = torch.tensor([focal_value(p) for p in [0.8, 0.7, 0.6, 0.6]], dtype=torch.float64)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


def test_softmax_composition_passes_numerical_gradient_check():
    logits = torch.tensor([[1.2, -0.4, 0.1], [-0.3, 0.2, 1.0]], dtype=torch.float64, requires_grad=True)
    labels = torch.tensor([0, 2])
    criterion = FocalLoss(apply_nonlin=lambda value: torch.softmax(value, dim=1), smooth=0.02)
    assert torch.autograd.gradcheck(lambda values: criterion(values, labels), (logits,), eps=1e-6, atol=1e-6)


def test_half_precision_inputs_accumulate_in_float32():
    probabilities = torch.tensor([[0.75, 0.25]], dtype=torch.float16, requires_grad=True)
    loss = FocalLoss(smooth=0)(probabilities, torch.tensor([0]))
    assert loss.dtype == torch.float32
    assert loss.item() == pytest.approx(focal_value(0.75), rel=1e-6)
    loss.backward()
    assert torch.isfinite(probabilities.grad).all()
