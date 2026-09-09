# ReAxis modifications, 2026-09-09: public naming, compatibility and release packaging.
"""ReAxis modules, named after Sections III-D and III-E of the paper.

This is a naming-only migration. Tensor operations, registered state-dict keys,
and the serialized ``starclip_modules``/``starclip_phase`` keys are unchanged.
"""


import math

import torch
from torch import nn
import torch.nn.functional as F


def _check_two_class(tensor, class_dim):
    if tensor.ndim < 2:
        raise ValueError("two-class tensor must have at least 2 dims, got {}".format(tuple(tensor.shape)))
    class_dim = class_dim if class_dim >= 0 else tensor.ndim + class_dim
    if class_dim < 0 or class_dim >= tensor.ndim:
        raise ValueError("invalid class_dim={} for shape {}".format(class_dim, tuple(tensor.shape)))
    if tensor.shape[class_dim] != 2:
        raise ValueError(
            "expected two-class dimension of size 2 at dim {}, got shape {}".format(
                class_dim,
                tuple(tensor.shape),
            )
        )
    return class_dim


def binary_margin(two_class_logits, class_dim=1):
    """Return abnormal-normal evidence margin from explicit two-class scores."""
    class_dim = _check_two_class(two_class_logits, class_dim)
    normal = two_class_logits.select(class_dim, 0)
    abnormal = two_class_logits.select(class_dim, 1)
    return abnormal - normal


def margin_to_two_class_logits(margin, class_dim=1):
    """Symmetric logits whose abnormal softmax probability equals sigmoid(margin)."""
    if margin.ndim == 0:
        raise ValueError("margin must include a batch dimension")
    class_dim = class_dim if class_dim >= 0 else margin.ndim + 1 + class_dim
    if class_dim < 0 or class_dim > margin.ndim:
        raise ValueError("invalid class_dim={} for margin shape {}".format(class_dim, tuple(margin.shape)))
    return torch.stack([-0.5 * margin, 0.5 * margin], dim=class_dim)


def probability_to_margin(probability, class_dim=1, eps=1e-6, return_sum_error=False):
    """Convert probabilities to abnormal-normal log-odds in fp32.

    For two-channel probability tensors this uses both channels:
    log(p_abnormal) - log(p_normal). Single-channel tensors are treated as
    abnormal probabilities only for legacy compatibility.
    """
    prob = probability.float()
    if prob.ndim > class_dim and prob.shape[class_dim] == 2:
        prob_sum = prob.sum(dim=class_dim, keepdim=True)
        sum_error = (prob_sum - 1.0).abs().max()
        prob = prob / prob_sum.clamp_min(eps)
        p_normal = prob.select(class_dim, 0).clamp_min(eps)
        p_abnormal = prob.select(class_dim, 1).clamp_min(eps)
        margin = torch.log(p_abnormal) - torch.log(p_normal)
        if return_sum_error:
            return margin, sum_error
        return margin

    prob = prob.clamp(eps, 1.0 - eps)
    margin = torch.logit(prob)
    if return_sum_error:
        return margin, torch.full((), float("nan"), device=prob.device)
    return margin


def normalize_two_class_probability(probability, class_dim=1, eps=1e-6):
    """Normalize an explicit two-channel probability tensor in fp32."""
    class_dim = _check_two_class(probability, class_dim)
    prob = probability.float()
    prob = prob.clamp_min(eps)
    prob = prob / prob.sum(dim=class_dim, keepdim=True).clamp_min(eps)
    return prob.to(dtype=probability.dtype)


def foreground_background_suppression(
    local_probability,
    guide_probability=None,
    strength=0.15,
    threshold=0.5,
    temperature=10.0,
    class_dim=1,
    eps=1e-6,
):
    """Conservatively suppress abnormal evidence in predicted background areas.

    The guide is normally the frozen Stage-1 fixed local prediction. It produces
    a soft foreground confidence from abnormal probability; low-confidence areas
    attenuate the target branch abnormal probability and the two channels are
    renormalized. This keeps the operation deterministic and target-label free.
    """
    if strength <= 0:
        return local_probability
    if strength > 1:
        raise ValueError("fb_suppression_strength must be in [0,1], got {}".format(strength))
    class_dim = _check_two_class(local_probability, class_dim)
    if class_dim != 1:
        raise ValueError("foreground_background_suppression expects class_dim=1 for local maps")

    prob = normalize_two_class_probability(local_probability, class_dim=class_dim, eps=eps).float()
    guide = prob if guide_probability is None else normalize_two_class_probability(
        guide_probability,
        class_dim=class_dim,
        eps=eps,
    ).float()
    if guide.shape != prob.shape:
        if guide.ndim != 4 or prob.ndim != 4:
            raise ValueError("guide and local probabilities must match shape or be [B,2,H,W]")
        guide = F.interpolate(guide, size=prob.shape[-2:], mode="bilinear", align_corners=False)
        guide = normalize_two_class_probability(guide, class_dim=class_dim, eps=eps).float()

    guide_abnormal = guide[:, 1:2]
    foreground_conf = torch.sigmoid(float(temperature) * (guide_abnormal - float(threshold)))
    attenuation = 1.0 - float(strength) * (1.0 - foreground_conf)
    abnormal = (prob[:, 1:2] * attenuation).clamp(eps, 1.0 - eps)
    normal = (1.0 - abnormal).clamp(eps, 1.0 - eps)
    suppressed = torch.cat([normal, abnormal], dim=1)
    suppressed = suppressed / suppressed.sum(dim=1, keepdim=True).clamp_min(eps)
    return suppressed.to(dtype=local_probability.dtype)


def _patch_tokens_and_grid(query_patch_feat):
    if query_patch_feat.ndim != 3:
        raise ValueError("query_patch_feat must be [B,N,D], got {}".format(tuple(query_patch_feat.shape)))
    num_tokens = query_patch_feat.shape[1]
    side_with_cls = int((num_tokens - 1) ** 0.5)
    if side_with_cls * side_with_cls == num_tokens - 1:
        return query_patch_feat[:, 1:, :], side_with_cls
    side = int(num_tokens ** 0.5)
    if side * side == num_tokens:
        return query_patch_feat, side
    raise ValueError("patch token count must be square with or without cls token, got {}".format(num_tokens))


def _resize_mask_to_patch_grid(mask, side):
    if mask.ndim == 4 and mask.shape[1] == 1:
        mask = mask[:, 0]
    if mask.ndim != 3:
        raise ValueError("mask must be [B,H,W] or [B,1,H,W], got {}".format(tuple(mask.shape)))
    return F.interpolate(
        mask.float().unsqueeze(1),
        size=(side, side),
        mode="bilinear",
        align_corners=False,
    )[:, 0]


def _evenly_subsample(features, max_count):
    if max_count is None or max_count <= 0 or features.shape[0] <= max_count:
        return features
    idx = torch.linspace(
        0,
        features.shape[0] - 1,
        steps=int(max_count),
        device=features.device,
    ).long()
    return features.index_select(0, idx)


def source_patch_memory_from_features(
    query_patch_feat,
    mask,
    max_normal_patches=4096,
    max_anomaly_patches=4096,
):
    """Extract normalized source normal/anomaly patch memory from labeled masks."""
    patch_tokens, side = _patch_tokens_and_grid(query_patch_feat)
    patch_tokens = F.normalize(patch_tokens.float(), dim=-1)
    mask_grid = _resize_mask_to_patch_grid(mask, side).flatten(1) > 0.5

    flat_tokens = patch_tokens.reshape(-1, patch_tokens.shape[-1])
    flat_mask = mask_grid.reshape(-1)
    normal = flat_tokens[~flat_mask]
    anomaly = flat_tokens[flat_mask]
    normal = _evenly_subsample(normal, max_normal_patches)
    anomaly = _evenly_subsample(anomaly, max_anomaly_patches)
    return normal.contiguous(), anomaly.contiguous()


def source_memory_anomaly_probability(
    query_patch_feat,
    normal_memory,
    anomaly_memory,
    topk=5,
    temperature=10.0,
    chunk_size=4096,
    output_size=None,
):
    """Score target patches by source-memory abnormal-vs-normal nearest evidence."""
    if normal_memory.numel() == 0 or anomaly_memory.numel() == 0:
        raise ValueError("source memory requires non-empty normal and anomaly patch banks")
    patch_tokens, side = _patch_tokens_and_grid(query_patch_feat)
    patch_tokens = F.normalize(patch_tokens.float(), dim=-1)
    normal_memory = F.normalize(normal_memory.float().to(patch_tokens.device), dim=-1)
    anomaly_memory = F.normalize(anomaly_memory.float().to(patch_tokens.device), dim=-1)

    flat_tokens = patch_tokens.reshape(-1, patch_tokens.shape[-1])
    topk_normal = max(1, min(int(topk), normal_memory.shape[0]))
    topk_anomaly = max(1, min(int(topk), anomaly_memory.shape[0]))
    chunk_size = max(1, int(chunk_size))
    scores = []
    for start in range(0, flat_tokens.shape[0], chunk_size):
        chunk = flat_tokens[start:start + chunk_size]
        sim_normal = chunk @ normal_memory.t()
        sim_anomaly = chunk @ anomaly_memory.t()
        normal_score = sim_normal.topk(k=topk_normal, dim=1).values.mean(dim=1)
        anomaly_score = sim_anomaly.topk(k=topk_anomaly, dim=1).values.mean(dim=1)
        scores.append(anomaly_score - normal_score)
    score = torch.cat(scores, dim=0).reshape(patch_tokens.shape[0], side, side)
    prob = torch.sigmoid(float(temperature) * score)
    if output_size is not None and tuple(prob.shape[-2:]) != tuple(output_size):
        prob = F.interpolate(
            prob.unsqueeze(1),
            size=tuple(output_size),
            mode="bilinear",
            align_corners=False,
        )[:, 0]
    return prob


def apply_source_memory_residual(
    local_probability,
    memory_probability,
    weight=0.1,
    class_dim=1,
    eps=1e-6,
):
    """Blend source-memory local probability into a two-channel local map."""
    if weight <= 0:
        return local_probability
    if weight > 1:
        raise ValueError("source_memory_weight must be in [0,1], got {}".format(weight))
    class_dim = _check_two_class(local_probability, class_dim)
    if class_dim != 1:
        raise ValueError("apply_source_memory_residual expects class_dim=1 for local maps")
    prob = normalize_two_class_probability(local_probability, class_dim=class_dim, eps=eps).float()
    memory = memory_probability.float()
    if memory.ndim == 4 and memory.shape[1] == 1:
        memory = memory[:, 0]
    if memory.ndim != 3:
        raise ValueError("memory_probability must be [B,H,W] or [B,1,H,W], got {}".format(tuple(memory.shape)))
    if tuple(memory.shape[-2:]) != tuple(prob.shape[-2:]):
        memory = F.interpolate(
            memory.unsqueeze(1),
            size=prob.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )[:, 0]
    abnormal = ((1.0 - float(weight)) * prob[:, 1] + float(weight) * memory).clamp(eps, 1.0 - eps)
    normal = (1.0 - abnormal).clamp(eps, 1.0 - eps)
    out = torch.stack([normal, abnormal], dim=1)
    out = out / out.sum(dim=1, keepdim=True).clamp_min(eps)
    return out.to(dtype=local_probability.dtype)


def bounded_residual(delta, cap):
    if cap <= 0:
        raise ValueError("residual cap must be positive, got {}".format(cap))
    return cap * torch.tanh(delta / cap)


def topk_mean(values, topk_ratio):
    if topk_ratio <= 0:
        raise ValueError("topk_ratio must be positive, got {}".format(topk_ratio))
    flat = values.reshape(values.shape[0], -1)
    k = max(1, int(flat.shape[1] * topk_ratio))
    return torch.topk(flat, k=k, dim=1, largest=True).values.mean(dim=1)


def local_image_probability(pixel_prob, mode="topk_mean", topk_ratio=0.01, logsumexp_temperature=0.1):
    flat = pixel_prob.reshape(pixel_prob.shape[0], -1)
    if mode == "max":
        return flat.max(dim=1).values
    if mode == "topk_mean":
        return topk_mean(pixel_prob, topk_ratio)
    if mode == "logsumexp":
        temp = max(float(logsumexp_temperature), 1e-6)
        return (temp * torch.logsumexp(flat / temp, dim=1) - temp * math.log(flat.shape[1])).clamp(0.0, 1.0)
    raise ValueError("unsupported image_local_pooling: {}".format(mode))


def _gaussian_kernel1d(sigma, dtype, device):
    radius = max(1, int(3.0 * float(sigma)))
    x = torch.arange(-radius, radius + 1, dtype=dtype, device=device)
    kernel = torch.exp(-(x ** 2) / (2.0 * float(sigma) ** 2))
    return kernel / kernel.sum().clamp_min(1e-12)


def gaussian_smoothing_2d(map_2d, sigma):
    """Device-independent separable Gaussian smoothing for [B,H,W] maps."""
    if sigma is None or float(sigma) <= 0:
        return map_2d
    if map_2d.ndim != 3:
        raise ValueError("gaussian_smoothing_2d expects [B,H,W], got {}".format(tuple(map_2d.shape)))
    dtype = map_2d.dtype
    device = map_2d.device
    kernel = _gaussian_kernel1d(float(sigma), dtype, device)
    pad = kernel.numel() // 2
    x = map_2d[:, None]
    kernel_x = kernel.view(1, 1, 1, -1)
    kernel_y = kernel.view(1, 1, -1, 1)
    pad_mode_x = "reflect" if map_2d.shape[-1] > pad else "replicate"
    pad_mode_y = "reflect" if map_2d.shape[-2] > pad else "replicate"
    x = F.pad(x, (pad, pad, 0, 0), mode=pad_mode_x)
    x = F.conv2d(x, kernel_x)
    x = F.pad(x, (0, 0, pad, pad), mode=pad_mode_y)
    x = F.conv2d(x, kernel_y)
    return x[:, 0]


def _infer_square_grid(num_tokens):
    side = int(num_tokens ** 0.5)
    if side * side != num_tokens:
        raise ValueError("patch token count must form a square grid, got {}".format(num_tokens))
    return side


# III-D.1: Query-Derived Normal Context (default output is h=[z; z_n; z-z_n]).
def build_query_derived_normal_context(
    query_global_feat,
    query_patch_feat,
    fixed_abnormal_prob,
    beta=10.0,
    source="global_plus_normal_context",
    detach_weights=True,
):
    """Build q, q_normal, and q-q_normal context without using ground-truth masks."""
    if query_global_feat.ndim != 2:
        raise ValueError("query_global_feat must be [B,D], got {}".format(tuple(query_global_feat.shape)))
    if query_patch_feat.ndim != 3:
        raise ValueError("query_patch_feat must be [B,N,D], got {}".format(tuple(query_patch_feat.shape)))
    if query_patch_feat.shape[0] != query_global_feat.shape[0]:
        raise ValueError("global and patch batch sizes differ")

    patch_tokens = query_patch_feat[:, 1:, :].float()
    side = _infer_square_grid(patch_tokens.shape[1])
    if fixed_abnormal_prob.ndim == 4 and fixed_abnormal_prob.shape[1] == 1:
        fixed_abnormal_prob = fixed_abnormal_prob[:, 0]
    if fixed_abnormal_prob.ndim == 3:
        prob_grid = F.interpolate(
            fixed_abnormal_prob.float().unsqueeze(1),
            size=(side, side),
            mode="bilinear",
            align_corners=False,
        ).flatten(1)
    elif fixed_abnormal_prob.ndim == 2:
        if fixed_abnormal_prob.shape[1] != patch_tokens.shape[1]:
            raise ValueError(
                "fixed abnormal probability length {} does not match patch tokens {}".format(
                    fixed_abnormal_prob.shape[1],
                    patch_tokens.shape[1],
                )
            )
        prob_grid = fixed_abnormal_prob.float()
    else:
        raise ValueError("fixed_abnormal_prob must be [B,H,W] or [B,N], got {}".format(tuple(fixed_abnormal_prob.shape)))

    if detach_weights:
        prob_grid = prob_grid.detach()
    weights = torch.softmax(-float(beta) * prob_grid, dim=1)
    q_normal = torch.einsum("bn,bnd->bd", weights, patch_tokens)
    q_global = F.normalize(query_global_feat.float(), dim=-1)
    q_normal = F.normalize(q_normal, dim=-1)

    if source == "global_only":
        return q_global
    if source == "normal_context_only":
        return q_normal
    if source == "global_plus_normal_context":
        return torch.cat([q_global, q_normal, q_global - q_normal], dim=-1)
    raise ValueError("unsupported anchor_condition_source: {}".format(source))


def _inverse_softplus(value):
    value = torch.as_tensor(float(value))
    return torch.log(torch.expm1(value).clamp_min(1e-12)).item()


def _logit(value):
    value = min(max(float(value), 1e-6), 1.0 - 1e-6)
    return math.log(value / (1.0 - value))


# III-E.1: Evidence Representation and Calibration.
class EvidenceCalibrator(nn.Module):
    """Per-branch scalar calibration: evidence = (margin - bias) / temperature."""

    def __init__(self, init_temperature=1.0, init_bias=0.0, eps=1e-6, temperature_min=0.05, temperature_max=20.0):
        super().__init__()
        self.eps = eps
        self.temperature_min = float(temperature_min)
        self.temperature_max = float(temperature_max)
        self.bias = nn.Parameter(torch.tensor(float(init_bias)))
        self.raw_temperature = nn.Parameter(torch.tensor(_inverse_softplus(float(init_temperature) - eps)))

    @property
    def temperature(self):
        temperature = F.softplus(self.raw_temperature) + self.eps
        return temperature.clamp(self.temperature_min, self.temperature_max)

    def forward(self, margin):
        return (margin - self.bias.to(dtype=margin.dtype, device=margin.device)) / self.temperature.to(
            dtype=margin.dtype,
            device=margin.device,
        )

    def stats(self):
        return {
            "temperature": float(self.temperature.detach().cpu().item()),
            "bias": float(self.bias.detach().cpu().item()),
        }

    def identity_regularization(self, bias_weight=1.0):
        temperature = self.temperature.float()
        return torch.log(temperature).pow(2) + float(bias_weight) * self.bias.float().pow(2)


# III-D.2: Structure-Preserving Axis Reorientation.
class StructurePreservingAxisReorientation(nn.Module):
    """Constrained contrast-direction residual updater for normal/abnormal anchors."""

    def __init__(
        self,
        anchor_dim=768,
        context_dim=None,
        hidden_dim=512,
        gamma_max=0.1,
        max_angle_deg=5.0,
        learnable_gamma=False,
        gamma_init=None,
        zero_init=True,
        center_eps=1e-6,
        angle_eps=1e-7,
    ):
        super().__init__()
        self.anchor_dim = int(anchor_dim)
        self.context_dim = int(context_dim or anchor_dim)
        self.gamma_max = float(gamma_max)
        self.max_angle_deg = float(max_angle_deg)
        self.max_angle_rad = math.radians(self.max_angle_deg)
        self.learnable_gamma = bool(learnable_gamma)
        self.center_eps = float(center_eps)
        self.angle_eps = float(angle_eps)
        gamma_init = self.max_angle_rad if gamma_init is None else float(gamma_init)
        gamma_init = min(max(gamma_init, 0.0), self.max_angle_rad)

        self.context_norm = nn.LayerNorm(self.context_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.context_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.anchor_dim),
        )
        if zero_init:
            nn.init.zeros_(self.mlp[-1].weight)
            nn.init.zeros_(self.mlp[-1].bias)
        if self.learnable_gamma:
            ratio = gamma_init / max(self.max_angle_rad, 1e-12)
            self.raw_gamma = nn.Parameter(torch.tensor(_logit(ratio)))
        else:
            self.register_buffer("fixed_angle_rad", torch.tensor(gamma_init))
        # Kept only so legacy checkpoints/configs can still report the old alias.
        self.register_buffer("fixed_gamma", torch.tensor(self.gamma_max))

    @property
    def gamma(self):
        if self.learnable_gamma:
            return self.max_angle_rad * torch.sigmoid(self.raw_gamma)
        return self.fixed_angle_rad

    @property
    def effective_max_angle_rad(self):
        return self.gamma

    def forward(self, context_feature, base_text_features):
        if context_feature.ndim != 2 or context_feature.shape[-1] != self.context_dim:
            raise ValueError(
                "context_feature must be [B,{}], got {}".format(self.context_dim, tuple(context_feature.shape))
            )
        if base_text_features.ndim == 2:
            if tuple(base_text_features.shape) != (2, self.anchor_dim):
                raise ValueError("base_text_features must be [2,D], got {}".format(tuple(base_text_features.shape)))
            base = base_text_features.unsqueeze(0).expand(context_feature.shape[0], -1, -1)
        elif base_text_features.ndim == 3:
            if base_text_features.shape[1:] != (2, self.anchor_dim):
                raise ValueError("base_text_features must be [B,2,D], got {}".format(tuple(base_text_features.shape)))
            if base_text_features.shape[0] != context_feature.shape[0]:
                raise ValueError("base_text_features and context batch sizes differ")
            base = base_text_features
        else:
            raise ValueError("base_text_features must be [2,D] or [B,2,D]")

        base = F.normalize(base.float(), dim=-1)
        t_normal = base[:, 0, :]
        t_abnormal = base[:, 1, :]
        center = 0.5 * (t_normal + t_abnormal)
        diff = 0.5 * (t_abnormal - t_normal)
        rho = diff.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        d_base = F.normalize(diff, dim=-1)

        delta_raw = self.mlp(self.context_norm(context_feature.float()))
        delta = delta_raw
        # Keep only a rotation residual orthogonal to the base contrast direction.
        delta = delta - (delta * d_base).sum(dim=-1, keepdim=True) * d_base

        center_norm = center.norm(dim=-1, keepdim=True)
        center_dir = center / center_norm.clamp_min(1e-6)
        center_mask = (center_norm > self.center_eps).float()
        delta = delta - center_mask * (delta * center_dir).sum(dim=-1, keepdim=True) * center_dir

        max_tangent = torch.tan(self.effective_max_angle_rad.to(device=delta.device, dtype=delta.dtype))
        delta_norm = delta.norm(dim=-1, keepdim=True)
        bounded_delta = max_tangent * delta / (1.0 + delta_norm)

        d_cond = F.normalize(d_base + bounded_delta, dim=-1)
        normal_cond = F.normalize(center - rho * d_cond, dim=-1)
        abnormal_cond = F.normalize(center + rho * d_cond, dim=-1)
        conditioned = torch.stack([normal_cond, abnormal_cond], dim=1)

        cos_rotation = (d_cond * d_base).sum(dim=-1).clamp(-1.0 + self.angle_eps, 1.0 - self.angle_eps)
        rotation_angle_rad = torch.acos(cos_rotation)
        rotation_angle_deg = rotation_angle_rad * (180.0 / math.pi)
        base_separation = (t_abnormal - t_normal).norm(dim=-1)
        conditioned_separation = (conditioned[:, 1, :] - conditioned[:, 0, :]).norm(dim=-1)
        return {
            "conditioned_anchors": conditioned,
            "delta_raw": delta_raw,
            "delta_projected": delta,
            "delta_bounded": bounded_delta,
            "d_base": d_base,
            "d_cond": d_cond,
            "anchor_raw_update_norm": delta_raw.norm(dim=-1),
            "anchor_update_norm": bounded_delta.norm(dim=-1),
            "anchor_rotation_angle": rotation_angle_deg,
            "anchor_rotation_angle_rad": rotation_angle_rad,
            "anchor_rotation_loss": rotation_angle_rad.pow(2),
            "anchor_center_alignment": (d_cond * center_dir).sum(dim=-1).abs(),
            "base_anchor_separation": base_separation,
            "conditioned_anchor_separation": conditioned_separation,
            "gamma": self.gamma,
            "max_angle_rad": self.effective_max_angle_rad,
            "max_angle_deg": self.effective_max_angle_rad * (180.0 / math.pi),
        }


# III-E.2: Hierarchical Residual Fusion (one component of CREF).
class HierarchicalResidualFusion(nn.Module):
    """Two-level residual fusion in calibrated evidence space."""

    def __init__(
        self,
        condition_residual_cap=4.0,
        global_residual_cap=4.0,
        local_residual_cap=4.0,
        condition_gate_init=0.15,
        adapt_global_gate_init=0.4,
        adapt_local_gate_init=0.4,
    ):
        super().__init__()
        self.condition_residual_cap = float(condition_residual_cap)
        self.global_residual_cap = float(global_residual_cap)
        self.local_residual_cap = float(local_residual_cap)
        self.condition_gate_init = float(condition_gate_init)
        self.adapt_global_gate_init = float(adapt_global_gate_init)
        self.adapt_local_gate_init = float(adapt_local_gate_init)
        self.raw_condition_local_gate = nn.Parameter(torch.tensor(_logit(condition_gate_init)))
        self.raw_adapt_global_gate = nn.Parameter(torch.tensor(_logit(adapt_global_gate_init)))
        self.raw_adapt_local_gate = nn.Parameter(torch.tensor(_logit(adapt_local_gate_init)))
        self.register_buffer("_condition_gate_init", torch.tensor(float(condition_gate_init)))
        self.register_buffer("_adapt_global_gate_init", torch.tensor(float(adapt_global_gate_init)))
        self.register_buffer("_adapt_local_gate_init", torch.tensor(float(adapt_local_gate_init)))

    def gates(self):
        return {
            "g_condition_local": torch.sigmoid(self.raw_condition_local_gate),
            "g_adapt_global": torch.sigmoid(self.raw_adapt_global_gate),
            "g_adapt_local": torch.sigmoid(self.raw_adapt_local_gate),
        }

    def gate_regularization(self, mode="initial"):
        gates = self.gates()
        if mode == "zero":
            return (
                gates["g_condition_local"].pow(2)
                + 0.1 * gates["g_adapt_global"].pow(2)
                + 0.1 * gates["g_adapt_local"].pow(2)
            )
        return (
            (gates["g_condition_local"] - self._condition_gate_init.to(gates["g_condition_local"].device)).pow(2)
            + (gates["g_adapt_global"] - self._adapt_global_gate_init.to(gates["g_adapt_global"].device)).pow(2)
            + (gates["g_adapt_local"] - self._adapt_local_gate_init.to(gates["g_adapt_local"].device)).pow(2)
        )

    def forward(self, e_fixed_global, e_fixed_local, e_base_global, e_base_local, e_cond_local):
        gates = self.gates()
        e_adapt_local = e_base_local + gates["g_condition_local"] * bounded_residual(
            e_cond_local - e_base_local,
            self.condition_residual_cap,
        )
        e_final_global = e_fixed_global + gates["g_adapt_global"] * bounded_residual(
            e_base_global - e_fixed_global,
            self.global_residual_cap,
        )
        e_final_local = e_fixed_local + gates["g_adapt_local"] * bounded_residual(
            e_adapt_local - e_fixed_local,
            self.local_residual_cap,
        )
        return e_adapt_local, e_final_global, e_final_local, gates


# III-E.3: Global-Local Anomaly Prediction.
class GlobalLocalAnomalyPrediction(nn.Module):
    """Independent image-level logit fusion from final global and raw local evidence."""

    def __init__(
        self,
        image_gate_init=0.5,
        image_bias_init=0.0,
        image_local_pooling="topk_mean",
        image_local_topk_ratio=0.01,
        image_local_pooling_space="native_grid",
        eps=1e-6,
    ):
        super().__init__()
        self.raw_image_gate = nn.Parameter(torch.tensor(_logit(image_gate_init)))
        self.image_bias = nn.Parameter(torch.tensor(float(image_bias_init)))
        self.image_gate_init = float(image_gate_init)
        self.image_local_pooling = image_local_pooling
        self.image_local_topk_ratio = float(image_local_topk_ratio)
        self.image_local_pooling_space = image_local_pooling_space
        self.eps = eps
        self.register_buffer("_image_gate_init", torch.tensor(float(image_gate_init)))

    @property
    def image_gate(self):
        return torch.sigmoid(self.raw_image_gate)

    def gate_regularization(self, mode="initial"):
        gate = self.image_gate
        if mode == "zero":
            return gate.pow(2)
        return (gate - self._image_gate_init.to(gate.device)).pow(2)

    def forward(self, e_final_global, e_final_local, gaussian_sigma=0, gaussian_for_image_score=False, pooling_size=None):
        local_evidence_for_pooling = e_final_local.float()
        if self.image_local_pooling_space == "native_grid" and pooling_size is not None:
            local_evidence_for_pooling = F.interpolate(
                local_evidence_for_pooling[:, None],
                size=pooling_size,
                mode="bilinear",
                align_corners=False,
            )[:, 0]
        pixel_prob_raw = torch.sigmoid(e_final_local.float())
        pixel_prob_for_pooling = torch.sigmoid(local_evidence_for_pooling)
        pooling_map = gaussian_smoothing_2d(pixel_prob_raw, gaussian_sigma) if gaussian_for_image_score else pixel_prob_raw
        if not gaussian_for_image_score:
            pooling_map = pixel_prob_for_pooling
        local_prob = local_image_probability(
            pooling_map,
            mode=self.image_local_pooling,
            topk_ratio=self.image_local_topk_ratio,
        )
        e_local_image = torch.logit(local_prob.clamp(self.eps, 1.0 - self.eps))
        gate = self.image_gate.to(device=e_final_global.device, dtype=e_final_global.dtype)
        image_bias = self.image_bias.to(device=e_final_global.device, dtype=e_final_global.dtype)
        image_logit = gate * e_final_global + (1.0 - gate) * e_local_image.to(e_final_global.dtype) + image_bias
        return {
            "pixel_prob_raw": pixel_prob_raw.to(e_final_local.dtype),
            "pixel_prob_eval": gaussian_smoothing_2d(pixel_prob_raw, gaussian_sigma).to(e_final_local.dtype),
            "local_image_probability": local_prob.to(e_final_global.dtype),
            "local_image_logit": e_local_image.to(e_final_global.dtype),
            "image_logit": image_logit,
            "image_score": torch.sigmoid(image_logit),
            "g_image_global_local": gate,
            "local_image_pooling_resolution": torch.tensor(pooling_map.shape[-2:], device=e_final_local.device),
        }


REAXIS_MODULES_KEY = "starclip_modules"
REAXIS_PHASE_KEY = "starclip_phase"
LEGACY_HPRF_MODULES_KEY = "hprf_modules"
LEGACY_STAGE22_PHASE_KEY = "stage22_phase"


class ReAxisStageIIModules(nn.Module):
    """ReAxis Stage II: axis reorientation and calibrated residual evidence fusion.

    Section III-D combines query-derived context with structure-preserving axis
    reorientation. Section III-E (CREF) comprises calibrators, hierarchical
    residual fusion and global-local anomaly prediction. Registered attribute
    names are retained for strict compatibility with existing checkpoints.
    """

    def __init__(
        self,
        input_dim=768,
        condition_source="global_plus_normal_context",
        hidden_dim=512,
        gamma_max=0.1,
        max_angle_deg=5.0,
        learnable_gamma=False,
        zero_init_anchor_updater=True,
        enable_branch_calibration=True,
        temperature_min=0.05,
        temperature_max=20.0,
        condition_residual_cap=4.0,
        global_residual_cap=4.0,
        local_residual_cap=4.0,
        condition_gate_init=0.15,
        adapt_global_gate_init=0.4,
        adapt_local_gate_init=0.4,
        image_gate_init=0.5,
        image_local_pooling="topk_mean",
        image_local_topk_ratio=0.01,
        image_local_pooling_space="native_grid",
    ):
        super().__init__()
        condition_dim = {
            "global_only": input_dim,
            "normal_context_only": input_dim,
            "global_plus_normal_context": input_dim * 3,
        }[condition_source]
        self.enable_branch_calibration = bool(enable_branch_calibration)
        self.anchor_updater = StructurePreservingAxisReorientation(
            anchor_dim=input_dim,
            context_dim=condition_dim,
            hidden_dim=hidden_dim,
            gamma_max=gamma_max,
            max_angle_deg=max_angle_deg,
            learnable_gamma=learnable_gamma,
            zero_init=zero_init_anchor_updater,
        )
        self.calibrator_fixed_global = EvidenceCalibrator(temperature_min=temperature_min, temperature_max=temperature_max)
        self.calibrator_adapt_global = EvidenceCalibrator(temperature_min=temperature_min, temperature_max=temperature_max)
        self.calibrator_fixed_local = EvidenceCalibrator(temperature_min=temperature_min, temperature_max=temperature_max)
        self.calibrator_adapt_local = EvidenceCalibrator(temperature_min=temperature_min, temperature_max=temperature_max)
        self.fusion = HierarchicalResidualFusion(
            condition_residual_cap=condition_residual_cap,
            global_residual_cap=global_residual_cap,
            local_residual_cap=local_residual_cap,
            condition_gate_init=condition_gate_init,
            adapt_global_gate_init=adapt_global_gate_init,
            adapt_local_gate_init=adapt_local_gate_init,
        )
        self.image_fusion = GlobalLocalAnomalyPrediction(
            image_gate_init=image_gate_init,
            image_local_pooling=image_local_pooling,
            image_local_topk_ratio=image_local_topk_ratio,
            image_local_pooling_space=image_local_pooling_space,
        )

    def calibrate(self, fixed_global, fixed_local, adapt_global, base_local, cond_local):
        if not self.enable_branch_calibration:
            return fixed_global, fixed_local, adapt_global, base_local, cond_local
        return (
            self.calibrator_fixed_global(fixed_global),
            self.calibrator_fixed_local(fixed_local),
            self.calibrator_adapt_global(adapt_global),
            self.calibrator_adapt_local(base_local),
            self.calibrator_adapt_local(cond_local),
        )

    def calibration_parameters_dict(self):
        return {
            "fixed_global": self.calibrator_fixed_global.stats(),
            "adapt_global": self.calibrator_adapt_global.stats(),
            "fixed_local": self.calibrator_fixed_local.stats(),
            "adapt_local": self.calibrator_adapt_local.stats(),
        }

    def scalar_gates_dict(self):
        gates = self.fusion.gates()
        gates["g_image_global_local"] = self.image_fusion.image_gate
        return gates

    def calibration_identity_regularization(self, bias_weight=1.0):
        if not self.enable_branch_calibration:
            return next(self.parameters()).new_zeros(())
        regs = [
            self.calibrator_fixed_global.identity_regularization(bias_weight),
            self.calibrator_adapt_global.identity_regularization(bias_weight),
            self.calibrator_fixed_local.identity_regularization(bias_weight),
            self.calibrator_adapt_local.identity_regularization(bias_weight),
        ]
        return torch.stack(regs).sum()

    def gate_regularization(self, mode="initial"):
        return self.fusion.gate_regularization(mode) + self.image_fusion.gate_regularization(mode)


def local_prob_two_class_from_margin(margin):
    return margin_to_two_class_logits(margin, class_dim=1).softmax(dim=1)


def legacy_linear_fusion(fixed_global_logit, fixed_local_score, adaptive_global_logit, adaptive_local_score, fuse_weight):
    return (
        fuse_weight * fixed_global_logit + (1.0 - fuse_weight) * adaptive_global_logit,
        fuse_weight * fixed_local_score + (1.0 - fuse_weight) * adaptive_local_score,
    )


def dual_anchor_margin_fusion(
    fixed_global_logit,
    fixed_local_score,
    adaptive_global_logit,
    adaptive_local_score,
    fuse_weight,
    mode="legacy_linear",
    base_global_logit=None,
    base_local_score=None,
    alpha_global=None,
    alpha_local=None,
):
    """Configurable dual-anchor evidence fusion.

    The original path fuses global logits and local probabilities directly.
    Margin modes convert both branches to abnormal-normal evidence before
    fusing, then return two-class logits/probabilities expected downstream.
    """
    if mode == "legacy_linear":
        return legacy_linear_fusion(
            fixed_global_logit,
            fixed_local_score,
            adaptive_global_logit,
            adaptive_local_score,
            fuse_weight,
        )

    alpha_g = float(fuse_weight if alpha_global is None else alpha_global)
    alpha_l = float(fuse_weight if alpha_local is None else alpha_local)
    if not (0.0 <= alpha_g <= 1.0):
        raise ValueError("alpha_global must be in [0,1], got {}".format(alpha_g))
    if not (0.0 <= alpha_l <= 1.0):
        raise ValueError("alpha_local must be in [0,1], got {}".format(alpha_l))

    if mode in ("global_base_local_conditioned", "margin_global_base_local_conditioned"):
        if base_global_logit is None or base_local_score is None:
            raise ValueError("{} requires base_global_logit and base_local_score".format(mode))

    if mode == "global_base_local_conditioned":
        return (
            alpha_g * fixed_global_logit + (1.0 - alpha_g) * base_global_logit,
            alpha_l * fixed_local_score + (1.0 - alpha_l) * adaptive_local_score,
        )

    if mode == "margin_fixed_conditioned":
        global_margin = (
            alpha_g * binary_margin(fixed_global_logit, class_dim=1)
            + (1.0 - alpha_g) * binary_margin(adaptive_global_logit, class_dim=1)
        )
        local_margin = (
            alpha_l * probability_to_margin(fixed_local_score, class_dim=1)
            + (1.0 - alpha_l) * probability_to_margin(adaptive_local_score, class_dim=1)
        )
    elif mode == "margin_global_base_local_conditioned":
        global_margin = (
            alpha_g * binary_margin(fixed_global_logit, class_dim=1)
            + (1.0 - alpha_g) * binary_margin(base_global_logit, class_dim=1)
        )
        local_margin = (
            alpha_l * probability_to_margin(fixed_local_score, class_dim=1)
            + (1.0 - alpha_l) * probability_to_margin(adaptive_local_score, class_dim=1)
        )
    else:
        raise ValueError("unsupported dual_anchor_fusion_mode: {}".format(mode))

    return (
        margin_to_two_class_logits(global_margin, class_dim=1),
        local_prob_two_class_from_margin(local_margin),
    )


def normalize_starclip_preset_name(preset):
    # Keep the historical return values for older experiment/config consumers.
    aliases = {
        "reaxis_full": "starclip_full",
        "reaxis_no_calibration": "starclip_no_calibration",
        "reaxis_conditioned_global_ablation": "starclip_conditioned_global_ablation",
        "legacy_stage22": "legacy_dual_anchor",
        "base_anchor_only": "base_dual_anchor_only",
        "conditioned_local_only": "conditioned_dual_anchor_local_only",
        "hprf_no_calibration": "starclip_no_calibration",
        "hprf_full": "starclip_full",
        "hprf_conditioned_global_ablation": "starclip_conditioned_global_ablation",
    }
    return aliases.get(preset, preset)


def apply_reaxis_preset(args):
    """Select ReAxis while retaining the legacy configuration fields."""
    preset = getattr(args, "reaxis_preset", None)
    if preset is None:
        preset = getattr(args, "starclip_preset", None)
    if preset is None:
        preset = getattr(args, "hprf_preset", "starclip_full")
    preset = normalize_starclip_preset_name(preset)
    args.starclip_preset = preset
    args.hprf_preset = preset
    args.reaxis_preset = normalize_reaxis_preset_name(preset)
    if preset == "legacy_dual_anchor":
        args.fusion_type = "legacy_linear"
        return args
    if preset == "base_dual_anchor_only":
        args.fusion_type = "base_anchor_only"
        args.enable_branch_calibration = False
        return args
    if preset == "conditioned_dual_anchor_local_only":
        args.fusion_type = "conditioned_local_only"
        args.conditioned_anchor_scope = "local"
        return args
    if preset == "starclip_no_calibration":
        args.fusion_type = "hierarchical_residual"
        args.enable_branch_calibration = False
        return args
    if preset == "starclip_full":
        args.fusion_type = "hierarchical_residual"
        args.conditioned_anchor_scope = "local"
        args.anchor_update_mode = "contrast_direction"
        args.anchor_condition_source = "global_plus_normal_context"
        args.enable_branch_calibration = True
        args.image_local_pooling = "topk_mean"
        args.gaussian_for_image_score = False
        args.image_score_fusion = "calibrated_logit_mix"
        return args
    if preset == "starclip_conditioned_global_ablation":
        args.fusion_type = "hierarchical_residual"
        args.conditioned_anchor_scope = "global_and_local"
        args.anchor_update_mode = "contrast_direction"
        return args
    raise ValueError("unsupported ReAxis preset: {}".format(preset))


def uses_reaxis(args):
    return getattr(args, "train_stage", None) == "stage2_visual_anchor_update" and getattr(
        args,
        "fusion_type",
        None,
    ) in {
        "hierarchical_residual",
        "base_anchor_only",
        "conditioned_local_only",
    }


HPRFStage22Modules = ReAxisStageIIModules
legacy_stage22_fusion = dual_anchor_margin_fusion
apply_hprf_preset = apply_reaxis_preset
uses_hprf = uses_reaxis


def normalize_reaxis_preset_name(preset):
    """Expose the paper framework name without changing legacy preset semantics."""
    preset = normalize_starclip_preset_name(preset)
    return {
        "starclip_full": "reaxis_full",
        "starclip_no_calibration": "reaxis_no_calibration",
        "starclip_conditioned_global_ablation": "reaxis_conditioned_global_ablation",
    }.get(preset, preset)


# Compatibility names: aliases, never duplicate nn.Module registrations.
STARCLIPDualAnchorModules = ReAxisStageIIModules
StructuredContrastAnchorUpdater = StructurePreservingAxisReorientation
BoundedResidualFusion = HierarchicalResidualFusion
GlobalLocalImageScoreFusion = GlobalLocalAnomalyPrediction
ScalarAffineCalibrator = EvidenceCalibrator
build_normal_context_feature = build_query_derived_normal_context
apply_starclip_preset = apply_reaxis_preset
uses_starclip = uses_reaxis
STARCLIP_MODULES_KEY = REAXIS_MODULES_KEY
STARCLIP_PHASE_KEY = REAXIS_PHASE_KEY
