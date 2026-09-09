# Modified for the ReAxis public release on 2026-09-09.
# FocalLoss uses a new probability-gather implementation of the focal-loss formula.
# Remaining helpers are retained from the AdaptCLIP-based research code.
# See LICENSE and THIRD_PARTY_NOTICES.md for source and licensing information.

from math import exp

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal loss for class probabilities, optionally transformed from logits.

    The channel dimension is 1. ``target`` contains class indices with one entry
    per sample or spatial position. Without ``apply_nonlin``, the caller must
    provide probabilities; this class does not implicitly apply a softmax.

    With smoothing s, the selected probability receives weight 1-s and the
    remaining probabilities receive weight s/(C-1), subject to the historical
    upper bound 1-s. An additional s is added to this weighted probability.
    Keeping that convention preserves the research configuration's loss.

    Focal loss: Lin et al., "Focal Loss for Dense Object Detection" (2017),
    https://arxiv.org/abs/1708.02002. This implementation uses indexed probability
    selection rather than constructing a dense one-hot target matrix.
    """

    def __init__(self, apply_nonlin=None, alpha=None, gamma=2, balance_index=0, smooth=1e-5, size_average=True):
        super().__init__()
        if smooth is not None and not 0 <= smooth <= 1:
            raise ValueError("smooth must lie between 0 and 1")
        self.apply_nonlin = apply_nonlin
        self.alpha = alpha
        self.gamma = gamma
        self.balance_index = balance_index
        self.smooth = smooth
        self.size_average = size_average

    def forward(self, logit, target):
        probabilities = self.apply_nonlin(logit) if self.apply_nonlin is not None else logit
        if probabilities.ndim < 2:
            raise ValueError("probabilities must have a class dimension at index 1")
        classes = probabilities.shape[1]
        rows = probabilities.movedim(1, -1).reshape(-1, classes)
        # Historical target weights are float32, including under mixed precision.
        rows = rows.to(dtype=torch.promote_types(rows.dtype, torch.float32))
        labels = target.to(device=rows.device, dtype=torch.long).reshape(-1)
        if labels.numel() != rows.shape[0]:
            raise ValueError("target must contain one class index per prediction")
        selected = rows.gather(1, labels[:, None]).squeeze(1)
        smoothing = 0.0 if self.smooth is None else self.smooth
        if smoothing:
            if classes < 2:
                raise ValueError("smoothed focal loss requires at least two classes")
            high = 1.0 - smoothing
            low = min(smoothing / (classes - 1), high)
            coefficients = torch.tensor([high, low], dtype=torch.float32, device=rows.device)
            other_mass = rows.sum(dim=1) - selected
            selected = selected * coefficients[0] + other_mass * coefficients[1]
        effective_probability = selected + smoothing

        if self.alpha is None:
            class_weights = torch.ones(classes, dtype=torch.float32, device=rows.device)
        elif isinstance(self.alpha, float):
            class_weights = torch.full((classes,), 1.0 - self.alpha, dtype=torch.float32, device=rows.device)
            class_weights[self.balance_index] = self.alpha
        elif isinstance(self.alpha, (list, np.ndarray)):
            class_weights = torch.as_tensor(self.alpha, dtype=torch.float32, device=rows.device).reshape(-1)
            if class_weights.numel() != classes:
                raise ValueError("alpha must have one weight per class")
            class_weights = class_weights / class_weights.sum()
        else:
            raise TypeError("alpha must be None, a float, a list, or a numpy array")
        sample_weights = class_weights[labels]
        focal_factor = (1.0 - effective_probability).pow(self.gamma)
        losses = -sample_weights * focal_factor * effective_probability.log()
        return losses.mean() if self.size_average else losses


class BinaryDiceLoss(nn.Module):
    def __init__(self):
        super(BinaryDiceLoss, self).__init__()

    def forward(self, input, targets):
        # 获取每个批次的大小 N
        N = targets.size()[0]
        # 平滑变量
        smooth = 1
        # 将宽高 reshape 到同一纬度
        input_flat = input.view(N, -1)
        targets_flat = targets.view(N, -1)

        intersection = input_flat * targets_flat
        N_dice_eff = (2 * intersection.sum(1) + smooth) / (input_flat.sum(1) + targets_flat.sum(1) + smooth)
        # 计算一个批次中平均每张图的损失
        loss = 1 - N_dice_eff.sum() / N
        return loss


def patch_score_alignment_loss(local_score, target, output_size=None):
    """PAL-style alignment over trainable two-channel local scores.

    Same-state patch scores are encouraged to be closer than normal-anomaly
    pairs. This keeps the PAL objective compatible with STAR-CLIP Stage II,
    where raw CLIP patch tokens are frozen.
    """
    if local_score.ndim != 4 or local_score.shape[1] != 2:
        raise ValueError("local_score must be [B, 2, H, W], got {}".format(tuple(local_score.shape)))
    if target.ndim == 3:
        target = target.unsqueeze(1)
    if target.ndim != 4:
        raise ValueError("target must be [B, H, W] or [B, 1, H, W], got {}".format(tuple(target.shape)))
    if output_size is not None:
        local_score = F.interpolate(local_score.float(), size=output_size, mode="bilinear", align_corners=False)
        target = F.interpolate(target.float(), size=output_size, mode="bilinear", align_corners=False)
    else:
        local_score = local_score.float()
        target = F.interpolate(target.float(), size=local_score.shape[-2:], mode="bilinear", align_corners=False)

    target = (target[:, 0] > 0.5)
    tokens = local_score.permute(0, 2, 3, 1).reshape(local_score.shape[0], -1, 2)
    tokens = F.normalize(tokens, dim=-1, eps=1e-6)
    masks = target.reshape(target.shape[0], -1)
    losses = []
    for token, mask in zip(tokens, masks):
        normal_count = int((~mask).sum().item())
        abnormal_count = int(mask.sum().item())
        if normal_count == 0 or abnormal_count == 0:
            continue
        signs = mask.float().mul(2.0).sub(1.0)
        pair_state = signs[:, None] * signs[None, :]
        sim = token @ token.t()
        same = sim[pair_state > 0]
        cross = sim[pair_state < 0]
        if same.numel() == 0 or cross.numel() == 0:
            continue
        losses.append(torch.relu(cross.mean() - same.mean()))
    if not losses:
        return local_score.new_zeros(())
    return torch.stack(losses).mean()


def semantic_consistency_regularization_loss(
    local_score,
    target,
    output_size=None,
    margin=0.5,
    eps=1e-6,
):
    """Foreground/background semantic consistency over local two-class scores.

    Source masks supervise the sign of abnormal-normal evidence at patch level:
    abnormal pixels should have positive margin, normal pixels negative margin.
    The loss is balanced between states when both exist, and remains valid for
    all-normal samples.
    """
    if local_score.ndim != 4 or local_score.shape[1] != 2:
        raise ValueError("local_score must be [B, 2, H, W], got {}".format(tuple(local_score.shape)))
    if target.ndim == 3:
        target = target.unsqueeze(1)
    if target.ndim != 4:
        raise ValueError("target must be [B, H, W] or [B, 1, H, W], got {}".format(tuple(target.shape)))
    if output_size is not None:
        local_score = F.interpolate(local_score.float(), size=output_size, mode="bilinear", align_corners=False)
        target = F.interpolate(target.float(), size=output_size, mode="bilinear", align_corners=False)
    else:
        local_score = local_score.float()
        target = F.interpolate(target.float(), size=local_score.shape[-2:], mode="bilinear", align_corners=False)

    prob = local_score.clamp_min(eps)
    prob = prob / prob.sum(dim=1, keepdim=True).clamp_min(eps)
    evidence = torch.log(prob[:, 1].clamp_min(eps)) - torch.log(prob[:, 0].clamp_min(eps))
    mask = target[:, 0] > 0.5
    margin = float(margin)

    losses = []
    if mask.any():
        losses.append(torch.relu(margin - evidence[mask]).mean())
    if (~mask).any():
        losses.append(torch.relu(evidence[~mask] + margin).mean())
    if not losses:
        return local_score.new_zeros(())
    return torch.stack(losses).mean()


def smooth(arr, lamda1):
    new_array = arr
    arr2 = torch.zeros_like(arr)
    arr2[:, :-1, :] = arr[:, 1:, :]
    arr2[:, -1, :] = arr[:, -1, :]

    new_array2 = torch.zeros_like(new_array)
    new_array2[:, :, :-1] = new_array[:, :, 1:]
    new_array2[:, :, -1] = new_array[:, :, -1]
    loss = (torch.sum((arr2 - arr) ** 2) + torch.sum((new_array2 - new_array) ** 2)) / 2
    return lamda1 * loss


def sparsity(arr, target, lamda2):
    if target == 0:
        loss = torch.mean(torch.norm(arr, dim=0))
    else:
        loss = torch.mean(torch.norm(1-arr, dim=0))
    return lamda2 * loss
