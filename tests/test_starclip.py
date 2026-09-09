import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from starcliplib.adaptclip import aggregate_patch_tokens_multiscale, compute_global_local_score_batchwise
from starcliplib.starclip import (
    GlobalLocalImageScoreFusion,
    STARCLIPDualAnchorModules,
    binary_margin,
    bounded_residual,
    build_normal_context_feature,
    legacy_linear_fusion,
    dual_anchor_margin_fusion,
    local_prob_two_class_from_margin,
    margin_to_two_class_logits,
    probability_to_margin,
    topk_mean,
)
from starcliplib.loss import patch_score_alignment_loss


def _unit(x):
    return F.normalize(x, dim=-1)


def test_margin_reconstruction_probability():
    margin = torch.linspace(-5, 5, steps=11)
    logits = margin_to_two_class_logits(margin)
    abnormal = logits.softmax(dim=1)[:, 1]
    assert torch.allclose(abnormal, torch.sigmoid(margin), atol=1e-6)
    assert torch.allclose(binary_margin(logits), margin, atol=1e-6)


def test_structured_updater_zero_init_preserves_base_and_norms():
    torch.manual_seed(1)
    base = _unit(torch.randn(2, 8))
    updater = STARCLIPDualAnchorModules(input_dim=8, hidden_dim=16, zero_init_anchor_updater=True).anchor_updater
    out = updater(torch.randn(3, 24), base)
    expected = base.unsqueeze(0).expand(3, -1, -1)
    assert torch.allclose(out["conditioned_anchors"], expected, atol=1e-5)
    assert torch.allclose(out["conditioned_anchors"].norm(dim=-1), torch.ones(3, 2), atol=1e-5)
    assert out["anchor_rotation_angle_rad"].max().item() < 1e-3
    assert torch.isfinite(out["conditioned_anchors"]).all()


def test_structured_updater_rotation_angle_bound_for_large_residual():
    torch.manual_seed(11)
    max_angle = 5.0
    base = _unit(torch.randn(2, 16))
    modules = STARCLIPDualAnchorModules(input_dim=16, hidden_dim=32, max_angle_deg=max_angle, zero_init_anchor_updater=False)
    updater = modules.anchor_updater
    with torch.no_grad():
        updater.mlp[-1].weight.fill_(1000.0)
        updater.mlp[-1].bias.fill_(1000.0)
    out = updater(torch.randn(4, 48), base)
    assert out["anchor_rotation_angle"].max().item() <= max_angle + 1e-3
    assert torch.allclose(out["conditioned_anchors"].norm(dim=-1), torch.ones(4, 2), atol=1e-5)


def test_conditioned_direction_center_orthogonal_and_separation_stable():
    torch.manual_seed(12)
    base = _unit(torch.randn(2, 16))
    updater = STARCLIPDualAnchorModules(input_dim=16, hidden_dim=32, zero_init_anchor_updater=False).anchor_updater
    out = updater(torch.randn(5, 48), base)
    assert out["anchor_center_alignment"].max().item() < 1e-4
    separation_delta = (out["conditioned_anchor_separation"] - out["base_anchor_separation"]).abs()
    assert separation_delta.max().item() < 1e-4


def test_two_channel_probability_to_margin_matches_log_odds():
    logits = torch.randn(2, 2, 3, 3)
    prob = logits.softmax(dim=1)
    margin = probability_to_margin(prob, class_dim=1)
    expected = torch.log(prob.float()[:, 1].clamp_min(1e-6)) - torch.log(prob.float()[:, 0].clamp_min(1e-6))
    assert torch.allclose(margin, expected, atol=1e-6)
    assert torch.allclose(margin, binary_margin(logits, class_dim=1), atol=1e-5)
    margin_half = probability_to_margin(prob.half(), class_dim=1)
    assert margin_half.dtype == torch.float32
    assert torch.isfinite(margin_half).all()


def test_gate_bounds_and_bounded_residual_cap():
    modules = STARCLIPDualAnchorModules(input_dim=8, hidden_dim=16)
    for gate in modules.scalar_gates_dict().values():
        assert 0.0 <= gate.item() <= 1.0
    residual = bounded_residual(torch.tensor([-100.0, 0.0, 100.0]), cap=4.0)
    assert residual.abs().max().item() <= 4.0


def test_topk_mean_keeps_at_least_one_element():
    values = torch.tensor([[0.1, 0.5, 0.2, 0.4]])
    assert torch.allclose(topk_mean(values, 0.001), torch.tensor([0.5]))


def test_image_topk_pooling_uses_raw_pre_gaussian_map():
    fusion = GlobalLocalImageScoreFusion(image_local_pooling="topk_mean", image_local_topk_ratio=0.25)
    global_e = torch.zeros(1)
    local_e = torch.full((1, 4, 4), -8.0)
    local_e[:, 0, 0] = 8.0
    out_raw = fusion(global_e, local_e, gaussian_sigma=4, gaussian_for_image_score=False)
    out_smoothed = fusion(global_e, local_e, gaussian_sigma=4, gaussian_for_image_score=True)
    assert out_raw["local_image_probability"].item() > out_smoothed["local_image_probability"].item()
    assert out_raw["local_image_pooling_resolution"].tolist() == [4, 4]


def test_patch_map_shape_and_no_broadcasting():
    torch.manual_seed(2)
    b, d, side = 2, 8, 4
    query = _unit(torch.randn(b, d))
    patches = [_unit(torch.randn(b, side * side + 1, d))]
    anchors = _unit(torch.randn(b, 2, d))
    global_logit, local_map = compute_global_local_score_batchwise(query, patches, anchors, img_size=8)
    assert global_logit.shape == (b, 2)
    assert local_map.shape == (b, 2, 8, 8)
    assert binary_margin(global_logit).shape == (b,)
    assert probability_to_margin(local_map, class_dim=1).shape == (b, 8, 8)


def test_patch_multiscale_aggregation_identity_and_shape():
    torch.manual_seed(21)
    patches = torch.randn(2, 17, 8)
    disabled = aggregate_patch_tokens_multiscale(patches, enabled=False)
    assert torch.allclose(disabled, patches)
    aggregated = aggregate_patch_tokens_multiscale(
        patches,
        enabled=True,
        kernel_sizes=(1, 3, 5),
        sigma=4.0,
        fuse="residual",
        residual_beta=0.1,
    )
    assert aggregated.shape == patches.shape
    assert torch.allclose(aggregated[:, :1], patches[:, :1])
    assert torch.isfinite(aggregated).all()


def test_patch_multiscale_score_shapes_remain_valid():
    torch.manual_seed(22)
    b, d, side = 2, 8, 4
    query = _unit(torch.randn(b, d))
    patches = [_unit(torch.randn(b, side * side + 1, d))]
    anchors = _unit(torch.randn(b, 2, d))
    global_logit, local_map = compute_global_local_score_batchwise(
        query,
        patches,
        anchors,
        img_size=8,
        patch_ms_agg=True,
        patch_ms_kernel_sizes=(1, 3, 5),
        patch_ms_sigma=4.0,
        patch_ms_fuse="residual",
        patch_ms_residual_beta=0.1,
    )
    assert global_logit.shape == (b, 2)
    assert local_map.shape == (b, 2, 8, 8)
    assert torch.isfinite(local_map).all()


def test_patch_score_alignment_empty_pairs_zero_and_grad_safe():
    logits = torch.randn(2, 2, 4, 4, requires_grad=True)
    local_score = logits.softmax(dim=1)
    normal_only_mask = torch.zeros(2, 4, 4)
    loss = patch_score_alignment_loss(local_score, normal_only_mask, output_size=(4, 4))
    assert loss.item() == 0.0
    mixed_mask = torch.zeros(2, 4, 4)
    mixed_mask[:, :2, :2] = 1.0
    loss = patch_score_alignment_loss(local_score, mixed_mask, output_size=(4, 4))
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None


def test_zero_residual_gates_return_calibrated_fixed():
    modules = STARCLIPDualAnchorModules(input_dim=8, hidden_dim=16)
    with torch.no_grad():
        modules.fusion.raw_adapt_global_gate.fill_(-50.0)
        modules.fusion.raw_adapt_local_gate.fill_(-50.0)
    fixed_global = torch.randn(2)
    fixed_local = torch.randn(2, 4, 4)
    base_global = torch.randn(2)
    base_local = torch.randn(2, 4, 4)
    cond_local = torch.randn(2, 4, 4)
    _, final_global, final_local, _ = modules.fusion(
        fixed_global,
        fixed_local,
        base_global,
        base_local,
        cond_local,
    )
    assert torch.allclose(final_global, fixed_global, atol=1e-5)
    assert torch.allclose(final_local, fixed_local, atol=1e-5)


def test_legacy_linear_fusion_matches_manual_formula():
    fixed_g = torch.randn(2, 2)
    fixed_l = torch.rand(2, 2, 4, 4)
    adapt_g = torch.randn(2, 2)
    adapt_l = torch.rand(2, 2, 4, 4)
    weight = 0.3
    fused_g, fused_l = legacy_linear_fusion(fixed_g, fixed_l, adapt_g, adapt_l, weight)
    assert torch.allclose(fused_g, weight * fixed_g + (1 - weight) * adapt_g)
    assert torch.allclose(fused_l, weight * fixed_l + (1 - weight) * adapt_l)


def test_configured_legacy_linear_fusion_keeps_old_formula():
    fixed_g = torch.randn(2, 2)
    fixed_l = torch.rand(2, 2, 4, 4)
    adapt_g = torch.randn(2, 2)
    adapt_l = torch.rand(2, 2, 4, 4)
    weight = 0.7
    fused_g, fused_l = dual_anchor_margin_fusion(
        fixed_g,
        fixed_l,
        adapt_g,
        adapt_l,
        weight,
        mode="legacy_linear",
    )
    assert torch.allclose(fused_g, weight * fixed_g + (1 - weight) * adapt_g)
    assert torch.allclose(fused_l, weight * fixed_l + (1 - weight) * adapt_l)


def test_global_base_local_conditioned_uses_independent_weights():
    fixed_g = torch.randn(2, 2)
    fixed_l = torch.rand(2, 2, 4, 4)
    cond_g = torch.randn(2, 2)
    cond_l = torch.rand(2, 2, 4, 4)
    base_g = torch.randn(2, 2)
    base_l = torch.rand(2, 2, 4, 4)
    fused_g, fused_l = dual_anchor_margin_fusion(
        fixed_g,
        fixed_l,
        cond_g,
        cond_l,
        0.5,
        mode="global_base_local_conditioned",
        base_global_logit=base_g,
        base_local_score=base_l,
        alpha_global=0.25,
        alpha_local=0.75,
    )
    assert torch.allclose(fused_g, 0.25 * fixed_g + 0.75 * base_g)
    assert torch.allclose(fused_l, 0.75 * fixed_l + 0.25 * cond_l)


def test_margin_fixed_conditioned_fusion_matches_weighted_log_odds():
    fixed_g = torch.randn(2, 2)
    cond_g = torch.randn(2, 2)
    fixed_l = torch.randn(2, 2, 3, 3).softmax(dim=1)
    cond_l = torch.randn(2, 2, 3, 3).softmax(dim=1)
    fused_g, fused_l = dual_anchor_margin_fusion(
        fixed_g,
        fixed_l,
        cond_g,
        cond_l,
        0.6,
        mode="margin_fixed_conditioned",
    )
    expected_g_margin = 0.6 * binary_margin(fixed_g) + 0.4 * binary_margin(cond_g)
    expected_l_margin = 0.6 * probability_to_margin(fixed_l) + 0.4 * probability_to_margin(cond_l)
    assert torch.allclose(binary_margin(fused_g), expected_g_margin, atol=1e-6)
    assert torch.allclose(fused_l[:, 1], torch.sigmoid(expected_l_margin), atol=1e-6)


def test_base_required_for_global_base_legacy_mode():
    fixed_g = torch.randn(2, 2)
    fixed_l = torch.rand(2, 2, 4, 4)
    cond_g = torch.randn(2, 2)
    cond_l = torch.rand(2, 2, 4, 4)
    with pytest.raises(ValueError):
        dual_anchor_margin_fusion(
            fixed_g,
            fixed_l,
            cond_g,
            cond_l,
            0.5,
            mode="global_base_local_conditioned",
        )


def test_starclip_synthetic_forward_backward_checkpoint_reload(tmp_path):
    torch.manual_seed(3)
    b, d, side, img_size = 2, 8, 4, 8
    modules = STARCLIPDualAnchorModules(input_dim=d, hidden_dim=16)
    optimizer = torch.optim.Adam(modules.parameters(), lr=1e-3)

    base = _unit(torch.randn(2, d))
    query = _unit(torch.randn(b, d))
    patches = [_unit(torch.randn(b, side * side + 1, d))]
    fixed_global = torch.randn(b, 2)
    fixed_local = torch.randn(b, 2, img_size, img_size).softmax(dim=1)
    base_global, base_local = compute_global_local_score_batchwise(query, patches, base.unsqueeze(0).expand(b, -1, -1), img_size)

    context = build_normal_context_feature(
        query,
        patches[-1],
        fixed_local[:, 1],
        beta=10.0,
        source="global_plus_normal_context",
    )
    updater_out = modules.anchor_updater(context, base)
    cond_global, cond_local = compute_global_local_score_batchwise(
        query,
        patches,
        updater_out["conditioned_anchors"],
        img_size,
    )

    fixed_global_m = binary_margin(fixed_global)
    fixed_local_m = probability_to_margin(fixed_local, class_dim=1)
    base_global_m = binary_margin(base_global)
    base_local_m = probability_to_margin(base_local, class_dim=1)
    cond_local_m = probability_to_margin(cond_local, class_dim=1)
    e_fixed_g, e_fixed_l, e_base_g, e_base_l, e_cond_l = modules.calibrate(
        fixed_global_m,
        fixed_local_m,
        base_global_m,
        base_local_m,
        cond_local_m,
    )
    adapt_l, final_g, final_l, gates = modules.fusion(e_fixed_g, e_fixed_l, e_base_g, e_base_l, e_cond_l)
    image_out = modules.image_fusion(final_g, final_l, gaussian_sigma=0)

    assert cond_global.shape == (b, 2)
    assert cond_local.shape == (b, 2, img_size, img_size)
    assert adapt_l.shape == (b, img_size, img_size)
    assert image_out["image_score"].shape == (b,)
    assert torch.isfinite(image_out["image_score"]).all()
    for gate in gates.values():
        assert 0.0 <= gate.item() <= 1.0

    # Inference path above does not require image labels or pixel masks.
    loss = image_out["image_logit"].mean() + local_prob_two_class_from_margin(final_l)[:, 1].mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    ckpt = tmp_path / "starclip.pt"
    torch.save({"starclip_modules": modules.state_dict(), "starclip_phase": "fusion"}, ckpt)
    reloaded = STARCLIPDualAnchorModules(input_dim=d, hidden_dim=16)
    reloaded.load_state_dict(torch.load(ckpt, map_location="cpu")["starclip_modules"])
