import torch
from types import SimpleNamespace

from starcliplib import (
    apply_source_memory_residual,
    source_memory_anomaly_probability,
    source_patch_memory_from_features,
)
from train import apply_batch_source_memory_for_training


def test_source_patch_memory_extracts_normal_and_anomaly_tokens():
    patch = torch.randn(2, 17, 8)
    mask = torch.zeros(2, 16, 16)
    mask[1, :8, :8] = 1

    normal, anomaly = source_patch_memory_from_features(patch, mask)

    assert normal.shape[1] == 8
    assert anomaly.shape[1] == 8
    assert normal.shape[0] > 0
    assert anomaly.shape[0] > 0
    assert torch.isfinite(normal).all()
    assert torch.isfinite(anomaly).all()


def test_source_memory_probability_shape_and_range():
    patch = torch.randn(3, 17, 8)
    normal = torch.randn(11, 8)
    anomaly = torch.randn(13, 8)

    prob = source_memory_anomaly_probability(
        patch,
        normal,
        anomaly,
        topk=3,
        temperature=5.0,
        output_size=(16, 16),
    )

    assert prob.shape == (3, 16, 16)
    assert torch.isfinite(prob).all()
    assert torch.all((prob >= 0) & (prob <= 1))


def test_source_memory_residual_preserves_probability_and_identity():
    local = torch.rand(2, 2, 8, 8)
    local = local / local.sum(dim=1, keepdim=True)
    memory = torch.rand(2, 8, 8)

    identity = apply_source_memory_residual(local, memory, weight=0.0)
    out = apply_source_memory_residual(local, memory, weight=0.25)

    assert torch.allclose(identity, local)
    assert out.shape == local.shape
    assert torch.allclose(out.sum(dim=1), torch.ones_like(out[:, 0]), atol=1e-6)
    assert torch.isfinite(out).all()


def test_training_batch_source_memory_keeps_shape_and_gradient():
    args = SimpleNamespace(
        use_source_memory_residual=True,
        source_memory_train_mode="batch",
        source_memory_max_normal_patches=64,
        source_memory_max_anomaly_patches=64,
        source_memory_topk=3,
        source_memory_temperature=5.0,
        source_memory_chunk_size=128,
        source_memory_weight=0.1,
    )
    patch = torch.randn(2, 17, 8)
    gt = torch.zeros(2, 16, 16)
    gt[1, :8, :8] = 1
    local = torch.rand(2, 2, 4, 4, requires_grad=True)
    local = local / local.sum(dim=1, keepdim=True)
    local.retain_grad()

    out, stats = apply_batch_source_memory_for_training(args, patch, gt, local)
    loss = out[:, 1].mean()
    loss.backward()

    assert out.shape == local.shape
    assert stats is not None
    assert stats["normal_patches"] > 0
    assert stats["anomaly_patches"] > 0
    assert local.grad is not None
    assert torch.isfinite(out).all()
    assert torch.isfinite(local.grad).all()
