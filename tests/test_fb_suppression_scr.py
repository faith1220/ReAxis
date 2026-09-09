import torch

from starcliplib import foreground_background_suppression
from starcliplib.loss import semantic_consistency_regularization_loss


def test_fb_suppression_identity_when_disabled():
    local = torch.rand(2, 2, 8, 8)
    local = local / local.sum(dim=1, keepdim=True)
    out = foreground_background_suppression(local, strength=0.0)
    assert torch.allclose(out, local)


def test_fb_suppression_preserves_two_channel_probability():
    local = torch.full((1, 2, 4, 4), 0.5)
    guide = torch.zeros_like(local)
    guide[:, 0] = 0.95
    guide[:, 1] = 0.05
    out = foreground_background_suppression(
        local,
        guide_probability=guide,
        strength=0.25,
        threshold=0.5,
        temperature=10.0,
    )
    assert out.shape == local.shape
    assert torch.allclose(out.sum(dim=1), torch.ones_like(out[:, 0]), atol=1e-6)
    assert torch.isfinite(out).all()
    assert torch.all(out[:, 1] <= local[:, 1] + 1e-6)


def test_scr_loss_is_finite_for_all_normal_and_mixed_masks():
    local = torch.rand(2, 2, 8, 8)
    local = local / local.sum(dim=1, keepdim=True)
    all_normal = torch.zeros(2, 8, 8)
    mixed = all_normal.clone()
    mixed[0, :2, :2] = 1

    loss_normal = semantic_consistency_regularization_loss(local, all_normal)
    loss_mixed = semantic_consistency_regularization_loss(local, mixed)

    assert loss_normal.ndim == 0
    assert loss_mixed.ndim == 0
    assert torch.isfinite(loss_normal)
    assert torch.isfinite(loss_mixed)
