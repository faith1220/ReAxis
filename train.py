# ReAxis modifications, 2026-09-09: public naming, compatibility and release packaging.
"""Training script for the ReAxis anomaly detection model."""

import argparse
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from tqdm import tqdm

import reaxislib
from adaptcliplib.reaxis import LEGACY_HPRF_MODULES_KEY, LEGACY_STAGE22_PHASE_KEY
from reaxislib import (BinaryDiceLoss, FocalLoss, PQAdapter, TextualAdapter,
                          ReAxisStageIIModules, VisionConditionedAnchorUpdater,
                          REAXIS_MODULES_KEY, REAXIS_PHASE_KEY,
                          VisualAdapter, apply_reaxis_preset, binary_margin,
                          apply_source_memory_residual,
                          build_query_derived_normal_context,
                          compute_global_local_score_batchwise,
                          foreground_background_suppression,
                          legacy_linear_fusion,
                          dual_anchor_margin_fusion,
                          local_prob_two_class_from_margin,
                          patch_score_alignment_loss,
                          probability_to_margin,
                          source_memory_anomaly_probability,
                          source_patch_memory_from_features,
                          semantic_consistency_regularization_loss,
                          uses_reaxis)
from dataset import Dataset, PromptDataset
from tools import (
    LINEAGE_KEY,
    build_checkpoint_lineage,
    build_source_validation_split,
    get_logger,
    get_transform,
    normalize,
    setup_seed,
    validate_parent_lineage,
)


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("yes", "true", "t", "1", "y"):
        return True
    if value in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def normalize_train_stage_name(train_stage):
    aliases = {
        "stage_i_source_supervised_base_evidence_learning": "stage2_learnable_anchor",
        "stage_ii_normality_guided_axis_reorientation": "stage2_visual_anchor_update",
        "stage_ii_normality_guided_axis_calibration": "stage2_visual_anchor_update",
        "stage_i_dual_anchor": "stage2_learnable_anchor",
        "stage_i_dual_anchor_boundary": "stage2_learnable_anchor",
        "stage_ii_visual_evidence": "stage2_visual_anchor_update",
        "stage_ii_dual_anchor_visual": "stage2_visual_anchor_update",
    }
    return aliases.get(train_stage, train_stage)


def set_requires_grad(module, requires_grad):
    for param in module.parameters():
        param.requires_grad_(requires_grad)


def log_trainable_parameters(logger, named_modules):
    trainable_names = []
    total_trainable = 0
    for module_name, module in named_modules:
        for name, param in module.named_parameters():
            if param.requires_grad:
                full_name = f"{module_name}.{name}"
                trainable_names.append(full_name)
                total_trainable += param.numel()
    logger.info("trainable parameter count: {}".format(total_trainable))
    for name in trainable_names:
        logger.info("trainable parameter: {}".format(name))
    return total_trainable


def log_static_anchor_debug(logger, static_text_features):
    with torch.no_grad():
        anchor_norm = static_text_features.norm(dim=-1)
        anchor_cos = F.cosine_similarity(
            static_text_features[0:1],
            static_text_features[1:2],
            dim=-1,
        ).item()
    logger.info("static_text_features shape: {}".format(tuple(static_text_features.shape)))
    logger.info("static_text_features norm: {}".format(anchor_norm.detach().cpu().tolist()))
    logger.info("cosine(static normal anchor, static abnormal anchor): {:.6f}".format(anchor_cos))


def log_stage1_batch_debug(logger, global_logit, local_score, total_loss, global_loss, local_loss):
    logger.info(
        "stage1 debug: visual_global_logits mean={:.6f}, max={:.6f}, min={:.6f}".format(
            global_logit.detach().float().mean().item(),
            global_logit.detach().float().max().item(),
            global_logit.detach().float().min().item(),
        )
    )
    logger.info(
        "stage1 debug: visual_local_map mean={:.6f}, max={:.6f}, min={:.6f}".format(
            local_score.detach().float().mean().item(),
            local_score.detach().float().max().item(),
            local_score.detach().float().min().item(),
        )
    )
    logger.info(
        "stage1 debug: total_loss={:.6f}, global_loss={:.6f}, local_loss={:.6f}".format(
            total_loss.detach().float().item(),
            global_loss.detach().float().item(),
            local_loss.detach().float().item(),
        )
    )


def log_tensor_stats(logger, name, tensor):
    tensor = tensor.detach().float()
    logger.info(
        "{} mean={:.6f}, max={:.6f}, min={:.6f}".format(
            name,
            tensor.mean().item(),
            tensor.max().item(),
            tensor.min().item(),
        )
    )


def infer_patch_grid_size(query_patch_feats):
    num_tokens = query_patch_feats[-1].shape[1] - 1
    side = int(round(np.sqrt(float(num_tokens))))
    if side * side != num_tokens:
        return None
    return (side, side)


def patch_ms_kwargs_for_branch(args, branch):
    if not getattr(args, "patch_ms_agg", False):
        return {}
    apply_to = getattr(args, "patch_ms_apply_to", "none")
    enabled = False
    if apply_to == "all_textual":
        enabled = True
    elif branch == "base" and apply_to in ("base", "base_and_conditioned"):
        enabled = True
    elif branch == "conditioned" and apply_to in ("conditioned", "base_and_conditioned"):
        enabled = True
    if not enabled:
        return {}
    return {
        "patch_ms_agg": True,
        "patch_ms_kernel_sizes": tuple(getattr(args, "patch_ms_kernel_sizes", [1, 3, 5])),
        "patch_ms_sigma": getattr(args, "patch_ms_sigma", 4.0),
        "patch_ms_fuse": getattr(args, "patch_ms_fuse", "residual"),
        "patch_ms_residual_beta": getattr(args, "patch_ms_residual_beta", 0.1),
    }


def fb_suppression_enabled_for_branch(args, branch):
    if not getattr(args, "use_fb_suppression", False):
        return False
    apply_to = getattr(args, "fb_suppression_apply_to", "none")
    if apply_to == "all_textual":
        return branch in ("base", "conditioned")
    if branch == "base" and apply_to in ("base", "base_and_conditioned"):
        return True
    if branch == "conditioned" and apply_to in ("conditioned", "base_and_conditioned"):
        return True
    if branch == "final" and apply_to == "final":
        return True
    return False


def maybe_apply_fb_suppression(args, branch, local_score, guide_score):
    if not fb_suppression_enabled_for_branch(args, branch):
        return local_score
    return foreground_background_suppression(
        local_score,
        guide_probability=guide_score,
        strength=args.fb_suppression_strength,
        threshold=args.fb_suppression_threshold,
        temperature=args.fb_suppression_temperature,
    )


def load_visual_adapter_state(visual_learner, state_dict, logger, source):
    incompatible = visual_learner.load_state_dict(state_dict, strict=False)
    logger.info("loaded VisualAdapter from {}".format(source))
    if incompatible.missing_keys:
        logger.info("VisualAdapter missing keys kept at initialization: {}".format(incompatible.missing_keys))
    if incompatible.unexpected_keys:
        logger.info("VisualAdapter unexpected keys ignored: {}".format(incompatible.unexpected_keys))


def keep_visual_lsar_trainable_if_requested(args, visual_learner, logger):
    if not getattr(args, "use_visual_lsar", False):
        return False
    if not getattr(args, "train_visual_lsar_when_frozen", True):
        return False
    visual_learner.set_visual_lsar_requires_grad(True)
    logger.info("VisualAdapter LSAR-lite residual parameters are trainable while base VisualAdapter is frozen")
    return True


def load_stage1_checkpoint_for_stage2(checkpoint_path, textual_learner, visual_learner, pq_learner, logger, args):
    if not checkpoint_path:
        raise ValueError("--stage1_checkpoint_path is required for train_stage=stage2_learnable_anchor")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError("Stage 1 checkpoint not found: {}".format(checkpoint_path))

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    validate_parent_lineage(
        args,
        checkpoint,
        checkpoint_path,
        logger=logger,
        allow_mismatch=args.allow_checkpoint_lineage_mismatch,
        require_known=args.source_validation_mode != "none" or args.source_validation_split_path is not None,
    )
    if "visual_learner" not in checkpoint:
        raise KeyError("Stage 1 checkpoint must contain key 'visual_learner': {}".format(checkpoint_path))

    load_visual_adapter_state(visual_learner, checkpoint["visual_learner"], logger, checkpoint_path)

    for key, module in [("textual_learner", textual_learner), ("pq_learner", pq_learner)]:
        if key not in checkpoint:
            logger.info("Stage 1 checkpoint has no key '{}'; keeping current initialization".format(key))
            continue
        incompatible = module.load_state_dict(checkpoint[key], strict=False)
        logger.info("loaded optional '{}' from Stage 1 checkpoint".format(key))
        if incompatible.missing_keys:
            logger.info("{} missing keys: {}".format(key, incompatible.missing_keys))
        if incompatible.unexpected_keys:
            logger.info("{} unexpected keys: {}".format(key, incompatible.unexpected_keys))


def load_stage_i_checkpoint_for_stage_ii(checkpoint_path, textual_learner, visual_learner, pq_learner, logger, args):
    if not checkpoint_path:
        raise ValueError("--stage2_checkpoint_path is required for train_stage=stage2_visual_anchor_update")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError("Stage I checkpoint not found: {}".format(checkpoint_path))

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    validate_parent_lineage(
        args,
        checkpoint,
        checkpoint_path,
        logger=logger,
        allow_mismatch=args.allow_checkpoint_lineage_mismatch,
        require_known=args.source_validation_mode != "none" or args.source_validation_split_path is not None,
    )
    missing_keys = [key for key in ["textual_learner", "visual_learner"] if key not in checkpoint]
    if missing_keys:
        raise KeyError("Stage I checkpoint missing keys {}: {}".format(missing_keys, checkpoint_path))

    textual_learner.load_state_dict(checkpoint["textual_learner"])
    load_visual_adapter_state(visual_learner, checkpoint["visual_learner"], logger, checkpoint_path)
    logger.info("loaded Stage I TextualAdapter and VisualAdapter from {}".format(checkpoint_path))

    if "pq_learner" in checkpoint:
        pq_learner.load_state_dict(checkpoint["pq_learner"], strict=False)
        logger.info("loaded optional 'pq_learner' from Stage I checkpoint")
    else:
        logger.info("Stage I checkpoint has no key 'pq_learner'; keeping current initialization")
    return checkpoint


def load_reaxis_modules_from_checkpoint(reaxis_modules, checkpoint, logger, source):
    if reaxis_modules is None:
        return
    if checkpoint is None:
        logger.info(
            "ReAxis modules not found in {}; using zero-initialized structured updater, "
            "identity calibration, and configured gate initializers".format(source)
        )
        return
    module_key = REAXIS_MODULES_KEY if REAXIS_MODULES_KEY in checkpoint else LEGACY_HPRF_MODULES_KEY
    if module_key not in checkpoint:
        logger.info(
            "ReAxis modules not found in {}; using zero-initialized structured updater, "
            "identity calibration, and configured gate initializers".format(source)
        )
        return
    incompatible = reaxis_modules.load_state_dict(checkpoint[module_key], strict=False)
    expected_missing_prefixes = (
        "anchor_updater.",
        "calibrator_",
        "fusion.",
        "image_fusion.",
    )
    unexpected = list(incompatible.unexpected_keys)
    missing = list(incompatible.missing_keys)
    if missing:
        logger.info("ReAxis expected missing keys kept at initialization: {}".format(missing))
    if unexpected:
        logger.info("ReAxis unexpected keys ignored with strong warning: {}".format(unexpected))
    non_expected = [key for key in missing if not key.startswith(expected_missing_prefixes)]
    if non_expected:
        logger.info("ReAxis non-standard missing keys: {}".format(non_expected))


def set_reaxis_train_phase(reaxis_modules, phase, args):
    set_requires_grad(reaxis_modules, False)
    set_requires_grad(reaxis_modules.anchor_updater, True)
    if phase == "fusion":
        set_requires_grad(reaxis_modules.fusion, True)
        set_requires_grad(reaxis_modules.image_fusion, True)
        if args.enable_branch_calibration:
            set_requires_grad(reaxis_modules.calibrator_fixed_global, True)
            set_requires_grad(reaxis_modules.calibrator_adapt_global, True)
            set_requires_grad(reaxis_modules.calibrator_fixed_local, True)
            set_requires_grad(reaxis_modules.calibrator_adapt_local, True)


def reaxis_trainable_parameter_groups(reaxis_modules, phase, args):
    groups = []
    updater_params = [p for p in reaxis_modules.anchor_updater.parameters() if p.requires_grad]
    if updater_params:
        groups.append({"params": updater_params, "lr": args.updater_lr, "name": "starclip_anchor_updater"})
    if phase == "fusion":
        fusion_params = []
        fusion_params += [p for p in reaxis_modules.fusion.parameters() if p.requires_grad]
        fusion_params += [p for p in reaxis_modules.image_fusion.parameters() if p.requires_grad]
        if fusion_params:
            groups.append({"params": fusion_params, "lr": args.fusion_lr, "name": "starclip_fusion"})
        calibration_params = []
        for calibrator in [
            reaxis_modules.calibrator_fixed_global,
            reaxis_modules.calibrator_adapt_global,
            reaxis_modules.calibrator_fixed_local,
            reaxis_modules.calibrator_adapt_local,
        ]:
            calibration_params += [p for p in calibrator.parameters() if p.requires_grad]
        if calibration_params:
            groups.append({"params": calibration_params, "lr": args.calibration_lr, "name": "starclip_calibration"})
    if not groups:
        raise ValueError("No trainable ReAxis parameters for phase={}".format(phase))
    return groups


def pixel_loss_from_margin(loss_focal, loss_dice, local_margin, gt):
    local_prob = local_prob_two_class_from_margin(local_margin)
    loss = loss_focal(local_prob, gt)
    loss = loss + loss_dice(local_prob[:, 1, :, :], gt)
    loss = loss + loss_dice(local_prob[:, 0, :, :], 1 - gt)
    return loss


def scalar_gate_regularization(reaxis_modules):
    return reaxis_modules.gate_regularization(mode="zero")


def gate_regularization(reaxis_modules, mode):
    return reaxis_modules.gate_regularization(mode=mode)


def resolve_stage_ii_warmup_epochs(args):
    if args.stage_ii_updater_warmup_epochs is not None:
        warmup_epochs = int(args.stage_ii_updater_warmup_epochs)
    else:
        warmup_epochs = int(np.ceil(float(args.epoch) * float(args.stage_ii_updater_warmup_ratio)))
    if args.epoch > 1:
        warmup_epochs = max(1, min(warmup_epochs, args.epoch - 1))
    else:
        warmup_epochs = 0
    return warmup_epochs


def normalize_stage_ii_lrs_and_warmup(args):
    if args.updater_lr is None:
        args.updater_lr = args.learning_rate
    if args.fusion_lr is None:
        args.fusion_lr = 0.1 * args.updater_lr
    if args.calibration_lr is None:
        args.calibration_lr = 0.1 * args.updater_lr
    args.stage_ii_resolved_warmup_epochs = resolve_stage_ii_warmup_epochs(args)
    return args


def reaxis_stage_ii_forward(
    args,
    reaxis_modules,
    query_feats,
    query_patch_feats,
    fixed_global_logit,
    fixed_local_map,
    base_text_features,
    base_global_logit,
    base_local_map,
    img_size,
):
    m_fixed_global = binary_margin(fixed_global_logit, class_dim=1)
    m_fixed_local, fixed_sum_error = probability_to_margin(fixed_local_map, class_dim=1, return_sum_error=True)
    m_base_global = binary_margin(base_global_logit, class_dim=1)
    m_base_local, base_sum_error = probability_to_margin(base_local_map, class_dim=1, return_sum_error=True)

    context_feature = build_query_derived_normal_context(
        query_feats.detach(),
        query_patch_feats[-1].detach(),
        fixed_local_map[:, 1, :, :].detach(),
        beta=args.normal_context_beta,
        source=args.anchor_condition_source,
        detach_weights=args.detach_normal_context_weights,
    )
    base_for_update = base_text_features.detach() if args.freeze_base_anchors_in_stage_ii else base_text_features
    updater_out = reaxis_modules.anchor_updater(context_feature, base_for_update)
    conditioned_anchors = updater_out["conditioned_anchors"]
    conditioned_global_logit, conditioned_local_map = compute_global_local_score_batchwise(
        query_feats.float(),
        [patch_feat.float() for patch_feat in query_patch_feats],
        conditioned_anchors,
        img_size,
    )
    m_conditioned_global = binary_margin(conditioned_global_logit, class_dim=1)
    m_conditioned_local, conditioned_sum_error = probability_to_margin(
        conditioned_local_map,
        class_dim=1,
        return_sum_error=True,
    )

    adapt_global_margin = (
        m_conditioned_global
        if args.conditioned_anchor_scope in ("global", "global_and_local")
        else m_base_global
    )
    (
        e_fixed_global,
        e_fixed_local,
        e_adapt_global,
        e_base_local,
        e_conditioned_local,
    ) = reaxis_modules.calibrate(
        m_fixed_global,
        m_fixed_local,
        adapt_global_margin,
        m_base_local,
        m_conditioned_local,
    )

    if args.fusion_type == "base_anchor_only":
        e_adaptive_local = e_base_local
        e_final_global = e_adapt_global
        e_final_local = e_base_local
        gates = reaxis_modules.fusion.gates()
    elif args.fusion_type == "conditioned_local_only":
        e_adaptive_local = e_conditioned_local
        e_final_global = e_adapt_global
        e_final_local = e_conditioned_local
        gates = reaxis_modules.fusion.gates()
    else:
        e_adaptive_local, e_final_global, e_final_local, gates = reaxis_modules.fusion(
            e_fixed_global,
            e_fixed_local,
            e_adapt_global,
            e_base_local,
            e_conditioned_local,
        )

    image_out = reaxis_modules.image_fusion(
        e_final_global,
        e_final_local,
        gaussian_sigma=args.sigma,
        gaussian_for_image_score=args.gaussian_for_image_score,
        pooling_size=(
            int((query_patch_feats[-1].shape[1] - 1) ** 0.5),
            int((query_patch_feats[-1].shape[1] - 1) ** 0.5),
        ),
    )
    if args.image_score_fusion == "legacy_arithmetic_average":
        legacy_score = 0.5 * (torch.sigmoid(e_final_global.float()) + image_out["local_image_probability"].float())
        image_out["image_score"] = legacy_score.to(e_final_global.dtype)
        image_out["image_logit"] = torch.logit(legacy_score.clamp(1e-6, 1.0 - 1e-6)).to(e_final_global.dtype)
    gates = dict(gates)
    gates["g_image_global_local"] = image_out["g_image_global_local"]

    return {
        "fixed_global_margin": m_fixed_global,
        "fixed_local_margin": m_fixed_local,
        "base_global_margin": m_base_global,
        "base_local_margin": m_base_local,
        "conditioned_global_margin": m_conditioned_global,
        "conditioned_local_margin": m_conditioned_local,
        "e_fixed_global": e_fixed_global,
        "e_fixed_local": e_fixed_local,
        "e_adapt_global": e_adapt_global,
        "e_base_local": e_base_local,
        "e_conditioned_local": e_conditioned_local,
        "adaptive_local_margin": e_adaptive_local,
        "final_global_margin": e_final_global,
        "final_local_margin": e_final_local,
        "pixel_prob_raw": image_out["pixel_prob_raw"],
        "pixel_prob_eval": image_out["pixel_prob_eval"],
        "local_image_probability": image_out["local_image_probability"],
        "local_image_logit": image_out["local_image_logit"],
        "image_logit": image_out["image_logit"],
        "image_score": image_out["image_score"],
        "conditioned_anchors": conditioned_anchors,
        "anchor_update_norm": updater_out["anchor_update_norm"],
        "anchor_raw_update_norm": updater_out["anchor_raw_update_norm"],
        "anchor_rotation_angle": updater_out["anchor_rotation_angle"],
        "anchor_rotation_angle_rad": updater_out["anchor_rotation_angle_rad"],
        "anchor_center_alignment": updater_out["anchor_center_alignment"],
        "base_anchor_separation": updater_out["base_anchor_separation"],
        "conditioned_anchor_separation": updater_out["conditioned_anchor_separation"],
        "anchor_rotation_loss": updater_out["anchor_rotation_loss"],
        "delta_raw": updater_out["delta_raw"],
        "delta_projected": updater_out["delta_projected"],
        "delta_bounded": updater_out["delta_bounded"],
        "probability_channel_sum_error": {
            "fixed_local": fixed_sum_error,
            "base_local": base_sum_error,
            "conditioned_local": conditioned_sum_error,
        },
        "local_image_pooling_resolution": image_out["local_image_pooling_resolution"],
        "fusion_gates": gates,
        "calibration_parameters": reaxis_modules.calibration_parameters_dict(),
        "fixed_global_logit": fixed_global_logit,
        "fixed_local_map": fixed_local_map,
        "base_global_logit": base_global_logit,
        "base_local_map": base_local_map,
        "conditioned_global_logit": conditioned_global_logit,
        "conditioned_local_map": conditioned_local_map,
    }


load_hprf_modules_from_checkpoint = load_reaxis_modules_from_checkpoint
set_hprf_train_phase = set_reaxis_train_phase
hprf_trainable_parameter_groups = reaxis_trainable_parameter_groups
resolve_stage22_warmup_epochs = resolve_stage_ii_warmup_epochs
normalize_stage22_lrs_and_warmup = normalize_stage_ii_lrs_and_warmup
hprf_stage22_forward = reaxis_stage_ii_forward
load_stage2_checkpoint_for_stage22 = load_stage_i_checkpoint_for_stage_ii


def compute_anchor_reg_loss(adaptive_text_features, static_text_features):
    static_text_features = static_text_features.detach()
    normal_reg = 1.0 - F.cosine_similarity(
        adaptive_text_features[0:1],
        static_text_features[0:1],
        dim=-1,
    ).mean()
    abnormal_reg = 1.0 - F.cosine_similarity(
        adaptive_text_features[1:2],
        static_text_features[1:2],
        dim=-1,
    ).mean()
    return normal_reg + abnormal_reg


def compute_updated_anchor_reg_loss(updated_text_features, static_text_features):
    static_text_features = static_text_features.detach().float()
    updated_text_features = updated_text_features.float()
    normal_reg = 1.0 - F.cosine_similarity(
        updated_text_features[:, 0, :],
        static_text_features[0:1].expand(updated_text_features.shape[0], -1),
        dim=-1,
    ).mean()
    abnormal_reg = 1.0 - F.cosine_similarity(
        updated_text_features[:, 1, :],
        static_text_features[1:2].expand(updated_text_features.shape[0], -1),
        dim=-1,
    ).mean()
    return normal_reg + abnormal_reg


def compute_delta_reg_loss(delta_normal, delta_abnormal):
    return delta_normal.float().pow(2).sum(dim=-1).mean() + delta_abnormal.float().pow(2).sum(dim=-1).mean()


def compute_topk_mean(anomaly_map, topk_ratio):
    anomaly_map = anomaly_map.reshape(anomaly_map.shape[0], -1)
    topk_count = max(1, int(anomaly_map.shape[1] * topk_ratio))
    return torch.topk(anomaly_map, k=topk_count, dim=1, largest=True).values.mean(dim=1)


def build_stage_ii_ranking_score(final_global_logit, final_local_map, score_source, topk_ratio):
    global_score = final_global_logit.softmax(-1)[:, 1]
    if score_source == "global":
        return global_score

    anomaly_map = final_local_map[:, 1]
    if score_source == "map_max":
        return anomaly_map.reshape(anomaly_map.shape[0], -1).max(dim=1).values
    if score_source == "global_plus_topk":
        topk_score = compute_topk_mean(anomaly_map, topk_ratio)
        return 0.5 * (global_score + topk_score)

    raise ValueError("Unsupported ranking_score_source: {}".format(score_source))


def compute_pairwise_ranking_loss(scores, labels, margin, max_pairs=0):
    labels = labels.to(scores.device).long()
    normal_scores = scores[labels == 0]
    abnormal_scores = scores[labels > 0]
    if normal_scores.numel() == 0 or abnormal_scores.numel() == 0:
        return scores.new_zeros(())

    pair_losses = F.relu(margin - (abnormal_scores[:, None] - normal_scores[None, :])).reshape(-1)
    if max_pairs is not None and max_pairs > 0 and pair_losses.numel() > max_pairs:
        pair_losses = torch.topk(pair_losses, k=max_pairs, largest=True).values
    return pair_losses.mean()


def compute_hard_normal_topk_loss(local_map, labels, topk_ratio, margin):
    labels = labels.to(local_map.device).long()
    normal_map = local_map[labels == 0, 1]
    if normal_map.numel() == 0:
        return local_map.new_zeros(())
    topk_score = compute_topk_mean(normal_map, topk_ratio)
    return F.relu(topk_score - margin).mean()


def apply_batch_source_memory_for_training(args, query_patch_feat, gt, local_map, logger=None, log_prefix=""):
    """Apply source-only batch memory residual during Stage II training.

    The memory is built from frozen CLIP patch features and source masks in the
    current training batch. It is detached, so gradients still train the local
    predictor through the residual-blended final map without updating CLIP.
    """
    if not getattr(args, "use_source_memory_residual", False):
        return local_map, None
    if getattr(args, "source_memory_train_mode", "none") != "batch":
        return local_map, None

    with torch.no_grad():
        normal_memory, anomaly_memory = source_patch_memory_from_features(
            query_patch_feat.detach(),
            gt.detach(),
            max_normal_patches=args.source_memory_max_normal_patches,
            max_anomaly_patches=args.source_memory_max_anomaly_patches,
        )
        if normal_memory.numel() == 0 or anomaly_memory.numel() == 0:
            if logger is not None:
                logger.info("{}source-memory train skipped: normal_patches={}, anomaly_patches={}".format(
                    log_prefix,
                    int(normal_memory.shape[0]) if normal_memory.ndim > 1 else 0,
                    int(anomaly_memory.shape[0]) if anomaly_memory.ndim > 1 else 0,
                ))
            return local_map, None
        memory_probability = source_memory_anomaly_probability(
            query_patch_feat.detach(),
            normal_memory,
            anomaly_memory,
            topk=args.source_memory_topk,
            temperature=args.source_memory_temperature,
            chunk_size=args.source_memory_chunk_size,
            output_size=local_map.shape[-2:],
        )

    local_map = apply_source_memory_residual(
        local_map,
        memory_probability,
        weight=args.source_memory_weight,
    )
    stats = {
        "normal_patches": int(normal_memory.shape[0]),
        "anomaly_patches": int(anomaly_memory.shape[0]),
        "memory_min": float(memory_probability.detach().float().min().item()),
        "memory_max": float(memory_probability.detach().float().max().item()),
        "memory_mean": float(memory_probability.detach().float().mean().item()),
    }
    return local_map, stats


def module_grad_norm(module):
    squared_norm = 0.0
    has_grad = False
    for param in module.parameters():
        if param.grad is None:
            continue
        has_grad = True
        squared_norm += param.grad.detach().float().norm(2).item() ** 2
    if not has_grad:
        return None
    return squared_norm ** 0.5


def named_param_grad_norm(module, param_name):
    param = dict(module.named_parameters()).get(param_name)
    if param is None or param.grad is None:
        return None
    return param.grad.detach().float().norm(2).item()


def log_stage2_batch_debug(
    logger,
    static_text_features,
    adaptive_text_features,
    fixed_global_logit,
    adaptive_global_logit,
    final_global_logit,
    fixed_local_map,
    adaptive_local_map,
    final_local_map,
    total_loss,
    global_loss,
    local_loss,
    anchor_reg_loss,
):
    with torch.no_grad():
        static_norm = static_text_features.detach().float().norm(dim=-1)
        adaptive_norm = adaptive_text_features.detach().float().norm(dim=-1)
        cos_static_adaptive_normal = F.cosine_similarity(
            static_text_features[0:1].detach().float(),
            adaptive_text_features[0:1].detach().float(),
            dim=-1,
        ).item()
        cos_static_adaptive_abnormal = F.cosine_similarity(
            static_text_features[1:2].detach().float(),
            adaptive_text_features[1:2].detach().float(),
            dim=-1,
        ).item()
        cos_adaptive_normal_abnormal = F.cosine_similarity(
            adaptive_text_features[0:1].detach().float(),
            adaptive_text_features[1:2].detach().float(),
            dim=-1,
        ).item()

    logger.info("stage2 debug: static_text_features shape: {}".format(tuple(static_text_features.shape)))
    logger.info("stage2 debug: static_text_features norm: {}".format(static_norm.detach().cpu().tolist()))
    logger.info("stage2 debug: adaptive_text_features shape: {}".format(tuple(adaptive_text_features.shape)))
    logger.info("stage2 debug: adaptive_text_features norm: {}".format(adaptive_norm.detach().cpu().tolist()))
    logger.info("stage2 debug: cos(static_normal, adaptive_normal): {:.6f}".format(cos_static_adaptive_normal))
    logger.info("stage2 debug: cos(static_abnormal, adaptive_abnormal): {:.6f}".format(cos_static_adaptive_abnormal))
    logger.info("stage2 debug: cos(adaptive_normal, adaptive_abnormal): {:.6f}".format(cos_adaptive_normal_abnormal))
    log_tensor_stats(logger, "stage2 debug: fixed_global_logit", fixed_global_logit)
    log_tensor_stats(logger, "stage2 debug: adaptive_global_logit", adaptive_global_logit)
    log_tensor_stats(logger, "stage2 debug: final_global_logit", final_global_logit)
    log_tensor_stats(logger, "stage2 debug: fixed_local_map", fixed_local_map)
    log_tensor_stats(logger, "stage2 debug: adaptive_local_map", adaptive_local_map)
    log_tensor_stats(logger, "stage2 debug: final_local_map", final_local_map)
    logger.info(
        "stage2 debug: total_loss={:.6f}, global_loss={:.6f}, local_loss={:.6f}, anchor_reg_loss={:.6f}".format(
            total_loss.detach().float().item(),
            global_loss.detach().float().item(),
            local_loss.detach().float().item(),
            anchor_reg_loss.detach().float().item(),
        )
    )


def log_stage_ii_batch_debug(
    logger,
    static_text_features,
    adaptive_text_features,
    updated_text_features,
    delta_normal,
    delta_abnormal,
    fixed_global_logit,
    adaptive_global_logit,
    final_global_logit,
    fixed_local_map,
    adaptive_local_map,
    final_local_map,
    total_loss,
    global_loss,
    local_loss,
    anchor_reg_loss,
    delta_reg_loss,
    consistency_loss,
):
    with torch.no_grad():
        static_features = static_text_features.detach().float()
        adaptive_features = adaptive_text_features.detach().float()
        updated_features = updated_text_features.detach().float()
        static_norm = static_features.norm(dim=-1)
        adaptive_norm = adaptive_features.norm(dim=-1)
        updated_norm = updated_features.norm(dim=-1).mean(dim=0)
        delta_normal_norm = delta_normal.detach().float().norm(dim=-1)
        delta_abnormal_norm = delta_abnormal.detach().float().norm(dim=-1)
        cos_static_adaptive_normal = F.cosine_similarity(static_features[0:1], adaptive_features[0:1], dim=-1).item()
        cos_static_adaptive_abnormal = F.cosine_similarity(static_features[1:2], adaptive_features[1:2], dim=-1).item()
        cos_static_updated_normal = F.cosine_similarity(
            static_features[0:1].expand(updated_features.shape[0], -1),
            updated_features[:, 0, :],
            dim=-1,
        ).mean().item()
        cos_static_updated_abnormal = F.cosine_similarity(
            static_features[1:2].expand(updated_features.shape[0], -1),
            updated_features[:, 1, :],
            dim=-1,
        ).mean().item()
        cos_updated_normal_abnormal = F.cosine_similarity(
            updated_features[:, 0, :],
            updated_features[:, 1, :],
            dim=-1,
        ).mean().item()

    logger.info("Stage II debug: train_stage=stage2_visual_anchor_update")
    logger.info("Stage II debug: static_text_features shape: {}".format(tuple(static_text_features.shape)))
    logger.info("Stage II debug: static_text_features norm: {}".format(static_norm.detach().cpu().tolist()))
    logger.info("Stage II debug: adaptive_text_features shape: {}".format(tuple(adaptive_text_features.shape)))
    logger.info("Stage II debug: adaptive_text_features norm: {}".format(adaptive_norm.detach().cpu().tolist()))
    logger.info("Stage II debug: updated_text_features shape: {}".format(tuple(updated_text_features.shape)))
    logger.info("Stage II debug: updated_text_features norm mean: {}".format(updated_norm.detach().cpu().tolist()))
    logger.info(
        "Stage II debug: delta_normal_norm mean={:.6f}, max={:.6f}".format(
            delta_normal_norm.mean().item(), delta_normal_norm.max().item()
        )
    )
    logger.info(
        "Stage II debug: delta_abnormal_norm mean={:.6f}, max={:.6f}".format(
            delta_abnormal_norm.mean().item(), delta_abnormal_norm.max().item()
        )
    )
    logger.info("Stage II debug: cos(static_normal, adaptive_normal): {:.6f}".format(cos_static_adaptive_normal))
    logger.info("Stage II debug: cos(static_abnormal, adaptive_abnormal): {:.6f}".format(cos_static_adaptive_abnormal))
    logger.info("Stage II debug: cos(static_normal, updated_normal) mean: {:.6f}".format(cos_static_updated_normal))
    logger.info("Stage II debug: cos(static_abnormal, updated_abnormal) mean: {:.6f}".format(cos_static_updated_abnormal))
    logger.info("Stage II debug: cos(updated_normal, updated_abnormal) mean: {:.6f}".format(cos_updated_normal_abnormal))
    log_tensor_stats(logger, "Stage II debug: fixed_global_logit", fixed_global_logit)
    log_tensor_stats(logger, "Stage II debug: adaptive_global_logit", adaptive_global_logit)
    log_tensor_stats(logger, "Stage II debug: final_global_logit", final_global_logit)
    log_tensor_stats(logger, "Stage II debug: fixed_local_map", fixed_local_map)
    log_tensor_stats(logger, "Stage II debug: adaptive_local_map", adaptive_local_map)
    log_tensor_stats(logger, "Stage II debug: final_local_map", final_local_map)
    logger.info(
        "Stage II debug: total_loss={:.6f}, global_loss={:.6f}, local_loss={:.6f}, "
        "anchor_reg_loss={:.6f}, delta_reg_loss={:.6f}, consistency_loss={:.6f}".format(
            total_loss.detach().float().item(),
            global_loss.detach().float().item(),
            local_loss.detach().float().item(),
            anchor_reg_loss.detach().float().item(),
            delta_reg_loss.detach().float().item(),
            consistency_loss.detach().float().item(),
        )
    )


def log_stage_ii_grad_debug(logger, anchor_updater, textual_learner, visual_learner):
    logger.info("Stage II debug: anchor_updater grad_norm: {}".format(module_grad_norm(anchor_updater)))
    logger.info("Stage II debug: textual_learner.ctx_pos grad_norm: {}".format(named_param_grad_norm(textual_learner, "ctx_pos")))
    logger.info("Stage II debug: textual_learner.ctx_neg grad_norm: {}".format(named_param_grad_norm(textual_learner, "ctx_neg")))
    logger.info("Stage II debug: visual_learner grad_norm: {}".format(module_grad_norm(visual_learner)))


def train(args):
    args.train_stage = normalize_train_stage_name(args.train_stage)
    img_size = args.image_size
    features_list = args.features_list
    save_path = args.save_path
    dataset_name = args.dataset
    batch_size = args.batch_size
    k_shots = args.k_shots
    seed = args.seed
    vl_reduction = args.vl_reduction
    pq_mid_dim = args.pq_mid_dim
    pq_context = args.pq_context
    stage1_fixed_anchor_visual = args.train_stage == "stage1_fixed_anchor_visual"
    stage2_learnable_anchor = args.train_stage == "stage2_learnable_anchor"
    stage2_visual_anchor_update = args.train_stage == "stage2_visual_anchor_update"
    stage2_anchor_stage = stage2_learnable_anchor or stage2_visual_anchor_update
    adapter_only_stage = stage1_fixed_anchor_visual or stage2_anchor_stage
    if stage2_visual_anchor_update:
        apply_reaxis_preset(args)
        normalize_stage_ii_lrs_and_warmup(args)
    reaxis_stage_ii = uses_reaxis(args)

    mode = 'train'

    log_file = f'{dataset_name}_{seed}seed_{k_shots}shot_{mode}_log.txt'
    logger = get_logger(args.save_path, log_file)

    logger.info('\n')
    logger.info(args)
    if args.source_validation_mode != "none" and args.source_validation_split_path is None:
        split_path, split_info = build_source_validation_split(
            args.train_data_path,
            dataset_name,
            mode=args.source_validation_mode,
            ratio=args.source_validation_ratio,
            seed=args.source_validation_seed,
            output_dir=os.path.join(args.save_path, "source_validation"),
            val_classes=args.source_validation_val_classes.split(",") if args.source_validation_val_classes else None,
        )
        args.source_validation_split_path = split_path
        logger.info("created source-validation split: {}".format(split_path))
        logger.info("source-validation split info: {}".format(split_info))
    elif args.source_validation_split_path is not None:
        logger.info("using source-validation split: {}".format(args.source_validation_split_path))
    if stage2_visual_anchor_update:
        logger.info(
            "ReAxis semantic role: Stage I = Source-Supervised Base Evidence Learning; "
            "Stage II = Normality-Guided Axis Reorientation; "
            "Evidence fusion = Calibrated Residual Evidence Fusion (CREF)"
        )
        logger.info("Stage II fusion_type={}, reaxis_preset={}".format(args.fusion_type, args.reaxis_preset))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # ====================== Model Initialization  ======================

    if args.pretrained_model == 'ViT-L/14@336px':
        model, _ = reaxislib.load(args.pretrained_model, device=device)
        DPAM_layer = 20
        patch_size = 14
        input_dim = 768
        model.visual.DAPM_replace(DPAM_layer = DPAM_layer)
    if args.pretrained_model == 'VITB16_PLUS_240':
        model, _ = reaxislib.load(args.pretrained_model, device=device)
        DPAM_layer = 10
        patch_size = 16
        input_dim = 640
        model.visual.DAPM_replace(DPAM_layer = DPAM_layer)

    if args.pretrained_model == 'ViT-L-14-CLIPA-336':
        model, _ = reaxislib.load(args.pretrained_model, device=device)
        DPAM_layer = 20
        patch_size = 14
        input_dim = 768
        model.visual.DAPM_replace(DPAM_layer = DPAM_layer)

    # ====================== Init Adapters ======================
    textual_learner = TextualAdapter(model.to("cpu"), img_size, args.n_ctx)
    visual_learner = VisualAdapter(
        img_size,
        patch_size,
        input_dim=input_dim,
        reduction=vl_reduction,
        use_visual_lsar=args.use_visual_lsar,
        visual_lsar_bottleneck_ratio=args.visual_lsar_bottleneck_ratio,
        visual_lsar_scale=args.visual_lsar_scale,
    )
    pq_learner = PQAdapter(img_size, patch_size, context=pq_context, input_dim=input_dim, mid_dim=pq_mid_dim, layers_num=len(features_list))
    anchor_updater = None
    reaxis_modules = None
    if stage2_visual_anchor_update and reaxis_stage_ii:
        reaxis_modules = ReAxisStageIIModules(
            input_dim=input_dim,
            condition_source=args.anchor_condition_source,
            hidden_dim=args.anchor_update_hidden_dim,
            gamma_max=args.anchor_update_gamma_max,
            max_angle_deg=args.anchor_update_max_angle_deg,
            learnable_gamma=args.learnable_anchor_update_gamma,
            zero_init_anchor_updater=args.zero_init_anchor_updater,
            enable_branch_calibration=args.enable_branch_calibration,
            temperature_min=args.calibration_temperature_min,
            temperature_max=args.calibration_temperature_max,
            condition_residual_cap=args.condition_residual_cap,
            global_residual_cap=args.global_residual_cap,
            local_residual_cap=args.local_residual_cap,
            condition_gate_init=args.condition_gate_init,
            adapt_global_gate_init=args.adapt_global_gate_init,
            adapt_local_gate_init=args.adapt_local_gate_init,
            image_gate_init=args.image_gate_init,
            image_local_pooling=args.image_local_pooling,
            image_local_topk_ratio=args.image_local_topk_ratio,
            image_local_pooling_space=args.image_local_pooling_space,
        )
    elif stage2_visual_anchor_update:
        anchor_updater = VisionConditionedAnchorUpdater(
            input_dim=input_dim,
            hidden_dim=args.anchor_update_hidden_dim,
            gamma=args.anchor_update_gamma,
        )

    model.to(device)
    textual_learner.to(device)
    visual_learner.to(device)
    pq_learner.to(device)
    if anchor_updater is not None:
        anchor_updater.to(device)
    if reaxis_modules is not None:
        reaxis_modules.to(device)

    if stage2_learnable_anchor:
        load_stage1_checkpoint_for_stage2(
            args.stage1_checkpoint_path,
            textual_learner,
            visual_learner,
            pq_learner,
            logger,
            args,
        )
    stage2_source_checkpoint = None
    start_epoch = 0
    resume_optimizer_state = None
    if stage2_visual_anchor_update:
        stage2_source_checkpoint = load_stage_i_checkpoint_for_stage_ii(
            args.stage2_checkpoint_path,
            textual_learner,
            visual_learner,
            pq_learner,
            logger,
            args,
        )
        if reaxis_modules is not None:
            load_reaxis_modules_from_checkpoint(
                reaxis_modules,
                stage2_source_checkpoint,
                logger,
                args.stage2_checkpoint_path,
            )
            if args.resume_stage_ii and (
                REAXIS_MODULES_KEY in stage2_source_checkpoint
                or LEGACY_HPRF_MODULES_KEY in stage2_source_checkpoint
            ):
                phase_key = REAXIS_PHASE_KEY if REAXIS_PHASE_KEY in stage2_source_checkpoint else LEGACY_STAGE22_PHASE_KEY
                start_epoch = int(stage2_source_checkpoint.get("epoch", 0))
                resume_optimizer_state = stage2_source_checkpoint.get("optimizer")
                logger.info(
                    "resume_stage_ii enabled: start_epoch={}, checkpoint_phase={}".format(
                        start_epoch,
                        stage2_source_checkpoint.get(phase_key, "unknown"),
                    )
                )

    model.eval()
    if stage1_fixed_anchor_visual:
        set_requires_grad(model, False)
        set_requires_grad(textual_learner, False)
        set_requires_grad(visual_learner, True)
        set_requires_grad(pq_learner, False)
        textual_learner.eval()
        visual_learner.train()
        pq_learner.eval()
        logger.info("train_stage=stage1_fixed_anchor_visual: only VisualAdapter is trainable")
        log_trainable_parameters(
            logger,
            [
                ("model", model),
                ("textual_learner", textual_learner),
                ("visual_learner", visual_learner),
                ("pq_learner", pq_learner),
            ],
        )
    elif stage2_learnable_anchor:
        set_requires_grad(model, False)
        set_requires_grad(textual_learner, True)
        set_requires_grad(visual_learner, not args.freeze_visual_adapter_in_stage2)
        set_requires_grad(pq_learner, False)
        visual_lsar_trainable = False
        if args.freeze_visual_adapter_in_stage2:
            visual_lsar_trainable = keep_visual_lsar_trainable_if_requested(args, visual_learner, logger)
        textual_learner.train()
        visual_learner.eval() if args.freeze_visual_adapter_in_stage2 else visual_learner.train()
        pq_learner.eval()
        logger.info(
            "train_stage=stage2_learnable_anchor: TextualAdapter prompt is trainable; "
            "VisualAdapter frozen={}, VisualAdapter LSAR-lite trainable={}".format(
                args.freeze_visual_adapter_in_stage2,
                visual_lsar_trainable,
            )
        )
        log_trainable_parameters(
            logger,
            [
                ("model", model),
                ("textual_learner", textual_learner),
                ("visual_learner", visual_learner),
                ("pq_learner", pq_learner),
            ],
        )
    elif stage2_visual_anchor_update:
        set_requires_grad(model, False)
        if reaxis_stage_ii and args.freeze_base_anchors_in_stage_ii:
            args.freeze_textual_adapter_in_stage_ii = True
        set_requires_grad(textual_learner, not args.freeze_textual_adapter_in_stage_ii)
        set_requires_grad(visual_learner, not args.freeze_visual_adapter_in_stage_ii)
        set_requires_grad(pq_learner, False)
        if reaxis_stage_ii:
            set_reaxis_train_phase(reaxis_modules, "updater_warmup", args)
        else:
            set_requires_grad(anchor_updater, True)
        visual_lsar_trainable = False
        if args.freeze_visual_adapter_in_stage_ii:
            visual_lsar_trainable = keep_visual_lsar_trainable_if_requested(args, visual_learner, logger)
        textual_learner.eval() if args.freeze_textual_adapter_in_stage_ii else textual_learner.train()
        visual_learner.eval() if args.freeze_visual_adapter_in_stage_ii else visual_learner.train()
        pq_learner.eval()
        if reaxis_stage_ii:
            reaxis_modules.train()
        else:
            anchor_updater.train()
        logger.info(
            "train_stage=stage2_visual_anchor_update: {} is trainable; "
            "TextualAdapter frozen={}, VisualAdapter frozen={}, VisualAdapter LSAR-lite trainable={}".format(
                "ReAxis dual-anchor modules" if reaxis_stage_ii else "legacy anchor_updater",
                args.freeze_textual_adapter_in_stage_ii,
                args.freeze_visual_adapter_in_stage_ii,
                visual_lsar_trainable,
            )
        )
        log_trainable_parameters(
            logger,
            [
                ("model", model),
                ("textual_learner", textual_learner),
                ("visual_learner", visual_learner),
                ("pq_learner", pq_learner),
                ("anchor_updater", anchor_updater if anchor_updater is not None else reaxis_modules),
            ],
        )
    else:
        textual_learner.train()
        visual_learner.train()
        pq_learner.train()

    textual_learner_parameters = sum(p.numel() for p in textual_learner.parameters())
    visual_learner_parameters = sum(p.numel() for p in visual_learner.parameters())
    pq_learner_parameters = sum(p.numel() for p in pq_learner.parameters())
    anchor_updater_parameters = sum(p.numel() for p in anchor_updater.parameters()) if anchor_updater is not None else 0
    reaxis_parameters = sum(p.numel() for p in reaxis_modules.parameters()) if reaxis_modules is not None else 0

    learned_parameters = textual_learner_parameters + visual_learner_parameters + pq_learner_parameters + anchor_updater_parameters + reaxis_parameters
    fixed_parameters = sum(p.numel() for p in model.parameters())


    print(f"textual_learner params:{(textual_learner_parameters):.0f}",
          f"visual_learner params:{(visual_learner_parameters)/1e+6:.1f}M",
          f"pq_learner params:{(pq_learner_parameters)/1e+6:.1f}M",
          f"anchor_updater params:{(anchor_updater_parameters)/1e+6:.1f}M",
          f"ReAxis params:{(reaxis_parameters)/1e+6:.1f}M",
          f"learned all parameters:{(learned_parameters)/1e+6:.1f}M",
          f"fixed params:{(fixed_parameters)/1e+6:.1f}M",
          f"all params:{(learned_parameters+fixed_parameters)/1e+6:.1f}M"
     )

    # ====================== Optimizer and Loss  ======================
    if stage1_fixed_anchor_visual:
        optimizer_parameters = list(filter(lambda p: p.requires_grad, visual_learner.parameters()))
    elif stage2_learnable_anchor:
        optimizer_parameters = list(filter(lambda p: p.requires_grad, textual_learner.parameters()))
        optimizer_parameters += list(filter(lambda p: p.requires_grad, visual_learner.parameters()))
        if len(optimizer_parameters) == 0:
            raise ValueError("No trainable parameters for train_stage=stage2_learnable_anchor")
    elif stage2_visual_anchor_update:
        if reaxis_stage_ii:
            current_reaxis_phase = "updater_warmup" if start_epoch < args.stage_ii_resolved_warmup_epochs else "fusion"
            set_reaxis_train_phase(reaxis_modules, current_reaxis_phase, args)
            optimizer_parameters = reaxis_trainable_parameter_groups(reaxis_modules, current_reaxis_phase, args)
            logger.info("ReAxis Stage II initial phase={}, warmup_epochs={}".format(
                current_reaxis_phase,
                args.stage_ii_resolved_warmup_epochs,
            ))
            log_trainable_parameters(logger, [("starclip_modules", reaxis_modules)])
        else:
            optimizer_parameters = list(filter(lambda p: p.requires_grad, anchor_updater.parameters()))
            if not args.freeze_textual_adapter_in_stage_ii:
                optimizer_parameters += list(filter(lambda p: p.requires_grad, textual_learner.parameters()))
            optimizer_parameters += list(filter(lambda p: p.requires_grad, visual_learner.parameters()))
            if len(optimizer_parameters) == 0:
                raise ValueError("No trainable parameters for train_stage=stage2_visual_anchor_update")
    else:
        optimizer_parameters = list(textual_learner.parameters()) + list(visual_learner.parameters()) + list(pq_learner.parameters())
    optimizer = torch.optim.Adam(
        optimizer_parameters,
        lr=args.learning_rate,
        betas=(0.5, 0.999)
        )
    if reaxis_stage_ii and resume_optimizer_state is not None:
        try:
            optimizer.load_state_dict(resume_optimizer_state)
            logger.info("restored ReAxis optimizer state from {}".format(args.stage2_checkpoint_path))
        except ValueError as exc:
            logger.info("could not restore optimizer state after phase/group change: {}".format(exc))

    loss_focal = FocalLoss()
    loss_dice = BinaryDiceLoss()


    # ====================== Data  ======================
    preprocess, target_transform = get_transform(image_size=args.image_size)
    train_data = Dataset(root=args.train_data_path, transform=preprocess, target_transform=target_transform, \
                         dataset_name = dataset_name, k_shots= k_shots, save_dir=save_path, mode='train', seed=seed,
                         source_split_path=args.source_validation_split_path,
                         source_split_role="train" if args.source_validation_split_path is not None else None)
    train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    obj_list = train_data.obj_list


    # ====================== forward and backward ======================
    textual_learner.prepare_static_text_feature(model)
    if adapter_only_stage:
        textual_learner.static_text_features = textual_learner.static_text_features.detach()
        log_static_anchor_debug(logger, textual_learner.static_text_features)
    for epoch in tqdm(range(start_epoch, args.epoch)):
        if reaxis_stage_ii:
            desired_phase = "updater_warmup" if epoch < args.stage_ii_resolved_warmup_epochs else "fusion"
            if desired_phase != current_reaxis_phase:
                current_reaxis_phase = desired_phase
                set_reaxis_train_phase(reaxis_modules, current_reaxis_phase, args)
                optimizer = torch.optim.Adam(
                    reaxis_trainable_parameter_groups(reaxis_modules, current_reaxis_phase, args),
                    betas=(0.5, 0.999),
                )
                logger.info("ReAxis Stage II switched to phase={}".format(current_reaxis_phase))
                log_trainable_parameters(logger, [("starclip_modules", reaxis_modules)])
        local_loss_list = []
        global_loss_list = []
        anchor_loss_list = []
        delta_loss_list = []
        consistency_loss_list = []
        ranking_loss_list = []
        hard_normal_loss_list = []
        patch_alignment_loss_list = []
        scr_loss_list = []
        source_memory_mean_list = []
        total_loss_list = []
        reaxis_loss_lists = {
            "cond_pixel": [],
            "adapt_pixel": [],
            "final_pixel": [],
            "final_global": [],
            "final_image": [],
            "rotation": [],
            "update": [],
            "gate": [],
            "calibration_identity": [],
        }
        reaxis_stat_lists = {
            "anchor_update_mean": [],
            "anchor_update_max": [],
            "anchor_rotation_mean": [],
            "anchor_rotation_max": [],
            "anchor_rotation_rad_mean": [],
            "anchor_rotation_rad_max": [],
        }

        for batch_idx, items in enumerate(tqdm(train_data_loader)):
            if args.max_train_batches is not None and batch_idx >= args.max_train_batches:
                logger.info("stopping epoch early after max_train_batches={}".format(args.max_train_batches))
                break
            if not adapter_only_stage:
                prompt_image = items['prompt_img'].to(device)  # B*s*c*h*w
                b, s, c, h, w = prompt_image.shape
                prompt_image = prompt_image.reshape(-1, c, h, w)

            image = items['img'].to(device)
            label =  items['anomaly']

            gt = items['img_mask'].squeeze().to(device)
            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0

            with torch.no_grad():
                query_feats, query_patch_feats = model.encode_image(image, args.features_list, DPAM_layer = DPAM_layer)
                if not adapter_only_stage:
                    prompt_feats, prompt_patch_feats = model.encode_image(prompt_image, args.features_list, DPAM_layer = DPAM_layer)
                    prompt_feats = prompt_feats.reshape(b, s, -1)
                    for idx in range(len(args.features_list)):
                        prompt_patch_feats[idx] = rearrange(prompt_patch_feats[idx], '(b s) l d -> b s l d', b=b, s=s)

            local_loss = 0
            global_loss = 0
            anchor_reg_loss = torch.zeros((), device=device)
            delta_reg_loss = torch.zeros((), device=device)
            consistency_loss = torch.zeros((), device=device)
            ranking_loss = torch.zeros((), device=device)
            hard_normal_loss = torch.zeros((), device=device)
            patch_alignment_loss = torch.zeros((), device=device)
            scr_loss = torch.zeros((), device=device)
            source_memory_stats = None

            # ====================== stage2: fixed + adaptive anchors ======================
            if stage2_learnable_anchor:
                static_text_features = textual_learner.static_text_features.detach()
                if args.freeze_visual_adapter_in_stage2 and not any(p.requires_grad for p in visual_learner.parameters()):
                    with torch.no_grad():
                        fixed_global_logit, fixed_local_map = visual_learner(query_feats, query_patch_feats, static_text_features)
                else:
                    fixed_global_logit, fixed_local_map = visual_learner(query_feats, query_patch_feats, static_text_features)

                learned_prompts, tokenized_prompts = textual_learner()
                adaptive_text_features = model.encode_text(learned_prompts, tokenized_prompts).float()
                adaptive_global_logit, adaptive_local_map = textual_learner.compute_global_local_score(
                    query_feats,
                    query_patch_feats,
                    adaptive_text_features,
                    **patch_ms_kwargs_for_branch(args, "base"),
                )
                adaptive_local_map = maybe_apply_fb_suppression(
                    args,
                    "base",
                    adaptive_local_map,
                    fixed_local_map.detach(),
                )

                fuse_weight = args.fixed_adaptive_fuse_weight
                final_global_logit = fuse_weight * fixed_global_logit + (1.0 - fuse_weight) * adaptive_global_logit
                final_local_map = fuse_weight * fixed_local_map + (1.0 - fuse_weight) * adaptive_local_map

                global_loss = F.cross_entropy(final_global_logit, label.long().to(device))
                local_loss = loss_focal(final_local_map, gt)
                local_loss += loss_dice(final_local_map[:, 1, :, :], gt)
                local_loss += loss_dice(final_local_map[:, 0, :, :], 1 - gt)
                if args.use_scr_loss and args.scr_loss_weight > 0:
                    scr_loss = semantic_consistency_regularization_loss(
                        adaptive_local_map,
                        gt,
                        margin=args.scr_margin,
                    )
                anchor_reg_loss = compute_anchor_reg_loss(adaptive_text_features, static_text_features)
                if args.use_scr_loss:
                    scr_loss_list.append(scr_loss.item())

            # ====================== Stage II: vision-conditioned adaptive anchors ======================
            if stage2_visual_anchor_update:
                static_text_features = textual_learner.static_text_features.detach()
                if args.freeze_visual_adapter_in_stage_ii and not any(p.requires_grad for p in visual_learner.parameters()):
                    with torch.no_grad():
                        fixed_global_logit, fixed_local_map = visual_learner(query_feats, query_patch_feats, static_text_features)
                else:
                    fixed_global_logit, fixed_local_map = visual_learner(query_feats, query_patch_feats, static_text_features)

                learned_prompts, tokenized_prompts = textual_learner()
                adaptive_text_features = model.encode_text(learned_prompts, tokenized_prompts).float()
                if args.freeze_textual_adapter_in_stage_ii:
                    adaptive_text_features = adaptive_text_features.detach()
                adaptive_text_features = F.normalize(adaptive_text_features, dim=-1)
                base_global_logit = None
                base_local_map = None
                legacy_needs_base = getattr(args, "dual_anchor_margin_fusion_mode", "legacy_linear") in (
                    "global_base_local_conditioned",
                    "margin_global_base_local_conditioned",
                )

                if reaxis_stage_ii:
                    base_global_logit, base_local_map = textual_learner.compute_global_local_score(
                        query_feats,
                        query_patch_feats,
                        adaptive_text_features,
                        **patch_ms_kwargs_for_branch(args, "base"),
                    )
                    reaxis_outputs = reaxis_stage_ii_forward(
                        args,
                        reaxis_modules,
                        query_feats,
                        query_patch_feats,
                        fixed_global_logit,
                        fixed_local_map,
                        adaptive_text_features,
                        base_global_logit,
                        base_local_map,
                        img_size,
                    )
                    label_float = label.float().to(device)
                    cond_pixel_loss = pixel_loss_from_margin(
                        loss_focal,
                        loss_dice,
                        reaxis_outputs["conditioned_local_margin"],
                        gt,
                    )
                    adapt_pixel_loss = pixel_loss_from_margin(
                        loss_focal,
                        loss_dice,
                        reaxis_outputs["adaptive_local_margin"],
                        gt,
                    )
                    final_pixel_loss = pixel_loss_from_margin(
                        loss_focal,
                        loss_dice,
                        reaxis_outputs["final_local_margin"],
                        gt,
                    )
                    final_global_loss = F.binary_cross_entropy_with_logits(
                        reaxis_outputs["final_global_margin"],
                        label_float,
                    )
                    final_image_loss = F.binary_cross_entropy_with_logits(
                        reaxis_outputs["image_logit"],
                        label_float,
                    )
                    rotation_loss = reaxis_outputs["anchor_rotation_loss"].mean()
                    update_loss = reaxis_outputs["delta_raw"].float().pow(2).sum(dim=-1).mean()
                    gate_loss = gate_regularization(reaxis_modules, args.gate_regularization_mode)
                    calibration_identity_loss = reaxis_modules.calibration_identity_regularization(
                        args.calibration_bias_regularization_ratio,
                    )

                    global_loss = final_global_loss
                    local_loss = final_pixel_loss
                    anchor_reg_loss = rotation_loss
                    delta_reg_loss = update_loss
                    consistency_loss = gate_loss
                else:
                    if legacy_needs_base:
                        base_global_logit, base_local_map = textual_learner.compute_global_local_score(
                            query_feats,
                            query_patch_feats,
                            adaptive_text_features,
                            **patch_ms_kwargs_for_branch(args, "base"),
                        )
                        base_local_map = maybe_apply_fb_suppression(
                            args,
                            "base",
                            base_local_map,
                            fixed_local_map.detach(),
                        )
                    updated_text_features, delta_normal, delta_abnormal = anchor_updater(
                        query_feats.detach().float(),
                        adaptive_text_features,
                    )
                    adaptive_global_logit, adaptive_local_map = compute_global_local_score_batchwise(
                        query_feats.float(),
                        [patch_feat.float() for patch_feat in query_patch_feats],
                        updated_text_features,
                        img_size,
                        **patch_ms_kwargs_for_branch(args, "conditioned"),
                    )
                    adaptive_local_map = maybe_apply_fb_suppression(
                        args,
                        "conditioned",
                        adaptive_local_map,
                        fixed_local_map.detach(),
                    )

                    fuse_weight = args.fixed_adaptive_fuse_weight
                    final_global_logit, final_local_map = dual_anchor_margin_fusion(
                        fixed_global_logit,
                        fixed_local_map,
                        adaptive_global_logit,
                        adaptive_local_map,
                        fuse_weight,
                        mode=args.dual_anchor_margin_fusion_mode,
                        base_global_logit=base_global_logit,
                        base_local_score=base_local_map,
                        alpha_global=args.legacy_alpha_global,
                        alpha_local=args.legacy_alpha_local,
                    )
                    final_local_map = maybe_apply_fb_suppression(
                        args,
                        "final",
                        final_local_map,
                        fixed_local_map.detach(),
                    )
                    final_local_map, source_memory_stats = apply_batch_source_memory_for_training(
                        args,
                        query_patch_feats[-1],
                        gt,
                        final_local_map,
                        logger=logger if len(local_loss_list) == 0 else None,
                        log_prefix="Stage II debug: ",
                    )

                    global_loss = F.cross_entropy(final_global_logit, label.long().to(device))
                    local_loss = loss_focal(final_local_map, gt)
                    local_loss += loss_dice(final_local_map[:, 1, :, :], gt)
                    local_loss += loss_dice(final_local_map[:, 0, :, :], 1 - gt)
                    if args.use_scr_loss and args.scr_loss_weight > 0:
                        scr_sources = []
                        if args.scr_loss_source in ("adaptive", "both"):
                            scr_sources.append(adaptive_local_map)
                        if args.scr_loss_source in ("final", "both"):
                            scr_sources.append(final_local_map)
                        if args.scr_loss_source == "base" and base_local_map is not None:
                            scr_sources.append(base_local_map)
                        if scr_sources:
                            scr_loss = torch.stack([
                                semantic_consistency_regularization_loss(
                                    source,
                                    gt,
                                    margin=args.scr_margin,
                                )
                                for source in scr_sources
                            ]).mean()
                    if args.use_patch_alignment_loss and args.patch_alignment_weight > 0:
                        patch_grid_size = infer_patch_grid_size(query_patch_feats)
                        alignment_sources = []
                        if args.patch_alignment_source in ("adaptive", "both"):
                            alignment_sources.append(adaptive_local_map)
                        if args.patch_alignment_source in ("final", "both"):
                            alignment_sources.append(final_local_map)
                        if alignment_sources:
                            patch_alignment_loss = torch.stack([
                                patch_score_alignment_loss(source, gt, output_size=patch_grid_size)
                                for source in alignment_sources
                            ]).mean()
                    anchor_reg_loss = compute_updated_anchor_reg_loss(updated_text_features, static_text_features)
                    delta_reg_loss = compute_delta_reg_loss(delta_normal, delta_abnormal)
                    consistency_loss = F.mse_loss(adaptive_local_map, fixed_local_map.detach())
                    if args.use_ranking_loss:
                        ranking_score = build_stage_ii_ranking_score(
                            final_global_logit,
                            final_local_map,
                            args.ranking_score_source,
                            args.ranking_topk_ratio,
                        )
                        ranking_loss = compute_pairwise_ranking_loss(
                            ranking_score,
                            label,
                            args.ranking_margin,
                            args.ranking_max_pairs,
                        )
                    if args.use_hard_normal_topk_loss:
                        hard_normal_map_source = {
                            "final": final_local_map,
                            "updated": adaptive_local_map,
                            "fixed": fixed_local_map,
                        }[args.hard_normal_source]
                        hard_normal_loss = compute_hard_normal_topk_loss(
                            hard_normal_map_source,
                            label,
                            args.hard_normal_topk_ratio,
                            args.hard_normal_margin,
                        )

            # ====================== visual_adapter ======================
            if (not stage2_anchor_stage) and (args.visual_learner or stage1_fixed_anchor_visual):
                static_text_features = textual_learner.static_text_features
                global_logit, local_score = visual_learner(query_feats, query_patch_feats, static_text_features)

                global_loss += F.cross_entropy(global_logit, label.long().to(device))

                local_loss += loss_focal(local_score, gt)
                local_loss += loss_dice(local_score[:, 1, :, :], gt)
                local_loss += loss_dice(local_score[:, 0, :, :], 1-gt)

            # ====================== textual_adapter ======================
            if args.textual_learner and not stage1_fixed_anchor_visual and not stage2_anchor_stage:
                learned_prompts, tokenized_prompts = textual_learner()
                learned_text_features = model.encode_text(learned_prompts, tokenized_prompts).float()  # [2, 768]
                global_logit, local_score = textual_learner.compute_global_local_score(query_feats, query_patch_feats, learned_text_features)

                global_loss += F.cross_entropy(global_logit, label.long().to(device))

                local_loss += loss_focal(local_score, gt)
                local_loss += loss_dice(local_score[:, 1, :, :], gt)
                local_loss += loss_dice(local_score[:, 0, :, :], 1-gt)

            # ====================== pq_adapter ======================
            if args.pq_learner and not stage1_fixed_anchor_visual and not stage2_anchor_stage:
                global_logit, local_score_list, align_score_list = pq_learner(query_feats, query_patch_feats, prompt_feats, prompt_patch_feats)

                for i in range(len(global_logit)):
                    global_loss += F.cross_entropy(global_logit[i], label.long().to(device))

                for i in range(len(local_score_list)):
                    local_loss += loss_focal(local_score_list[i], gt)
                    local_loss += loss_dice(local_score_list[i][:, 1, :, :], gt)
                    local_loss += loss_dice(local_score_list[i][:, 0, :, :], 1-gt)


            optimizer.zero_grad()
            if stage2_learnable_anchor:
                total_loss = (
                    local_loss
                    + global_loss
                    + args.anchor_reg_weight * anchor_reg_loss
                    + args.scr_loss_weight * scr_loss
                )
                if len(local_loss_list) == 0:
                    log_stage2_batch_debug(
                        logger,
                        textual_learner.static_text_features,
                        adaptive_text_features,
                        fixed_global_logit,
                        adaptive_global_logit,
                        final_global_logit,
                        fixed_local_map,
                        adaptive_local_map,
                        final_local_map,
                        total_loss,
                        global_loss,
                        local_loss,
                        anchor_reg_loss,
                    )
            elif stage2_visual_anchor_update:
                if reaxis_stage_ii:
                    if current_reaxis_phase == "updater_warmup":
                        total_loss = (
                            args.loss_cond_pixel_weight * cond_pixel_loss
                            + args.loss_adapt_pixel_weight * adapt_pixel_loss
                            + args.loss_anchor_rotation_weight * rotation_loss
                            + args.loss_anchor_update_weight * update_loss
                            + args.loss_gate_regularization_weight * gate_loss
                            + args.loss_calibration_identity_weight * calibration_identity_loss
                        )
                    else:
                        total_loss = (
                            args.loss_cond_pixel_weight * cond_pixel_loss
                            + args.loss_adapt_pixel_weight * adapt_pixel_loss
                            + args.loss_final_pixel_weight * final_pixel_loss
                            + args.loss_final_global_weight * final_global_loss
                            + args.loss_final_image_weight * final_image_loss
                            + args.loss_anchor_rotation_weight * rotation_loss
                            + args.loss_anchor_update_weight * update_loss
                            + args.loss_gate_regularization_weight * gate_loss
                            + args.loss_calibration_identity_weight * calibration_identity_loss
                        )
                    if len(local_loss_list) == 0:
                        logger.info("ReAxis Stage II debug: phase={}".format(current_reaxis_phase))
                        for key in [
                            "fixed_global_margin",
                            "base_global_margin",
                            "conditioned_global_margin",
                            "final_global_margin",
                            "fixed_local_margin",
                            "base_local_margin",
                            "conditioned_local_margin",
                            "final_local_margin",
                        ]:
                            value = reaxis_outputs[key].detach().float()
                            logger.info(
                                "ReAxis Stage II debug: {} mean={:.6f}, std={:.6f}, min={:.6f}, max={:.6f}".format(
                                    key,
                                    value.mean().item(),
                                    value.std(unbiased=False).item(),
                                    value.min().item(),
                                    value.max().item(),
                                )
                            )
                        logger.info(
                            "ReAxis Stage II debug: losses cond_pixel={:.6f}, adapt_pixel={:.6f}, "
                            "final_pixel={:.6f}, final_global={:.6f}, final_image={:.6f}, "
                            "rotation={:.6f}, update={:.6f}, gate={:.6f}, cal_identity={:.6f}, total={:.6f}".format(
                                cond_pixel_loss.detach().float().item(),
                                adapt_pixel_loss.detach().float().item(),
                                final_pixel_loss.detach().float().item(),
                                final_global_loss.detach().float().item(),
                                final_image_loss.detach().float().item(),
                                rotation_loss.detach().float().item(),
                                update_loss.detach().float().item(),
                                gate_loss.detach().float().item(),
                                calibration_identity_loss.detach().float().item(),
                                total_loss.detach().float().item(),
                            )
                        )
                        logger.info(
                            "ReAxis Stage II debug: gates {}".format(
                                {
                                    key: float(value.detach().cpu().item())
                                    for key, value in reaxis_outputs["fusion_gates"].items()
                                }
                            )
                        )
                        logger.info(
                            "ReAxis Stage II debug: calibration {}".format(
                                reaxis_outputs["calibration_parameters"]
                            )
                        )
                        logger.info(
                            "ReAxis Stage II debug: anchor_update_norm mean={:.6f}, max={:.6f}; "
                            "anchor_rotation_angle_deg mean={:.6f}, max={:.6f}; "
                            "anchor_rotation_angle_rad mean={:.6f}, max={:.6f}; pooling_resolution={}".format(
                                reaxis_outputs["anchor_update_norm"].detach().float().mean().item(),
                                reaxis_outputs["anchor_update_norm"].detach().float().max().item(),
                                reaxis_outputs["anchor_rotation_angle"].detach().float().mean().item(),
                                reaxis_outputs["anchor_rotation_angle"].detach().float().max().item(),
                                reaxis_outputs["anchor_rotation_angle_rad"].detach().float().mean().item(),
                                reaxis_outputs["anchor_rotation_angle_rad"].detach().float().max().item(),
                                reaxis_outputs["local_image_pooling_resolution"].detach().cpu().tolist(),
                            )
                        )
                        logger.info(
                            "ReAxis Stage II debug: probability channel sum max error {}".format(
                                {
                                    key: float(value.detach().cpu().item())
                                    for key, value in reaxis_outputs["probability_channel_sum_error"].items()
                                }
                            )
                        )
                else:
                    total_loss = (
                        local_loss
                        + global_loss
                        + args.anchor_reg_weight * anchor_reg_loss
                        + args.delta_reg_weight * delta_reg_loss
                        + args.anchor_update_cons_weight * consistency_loss
                        + args.ranking_weight * ranking_loss
                        + args.hard_normal_weight * hard_normal_loss
                        + args.patch_alignment_weight * patch_alignment_loss
                        + args.scr_loss_weight * scr_loss
                    )
                    if len(local_loss_list) == 0:
                        log_stage_ii_batch_debug(
                            logger,
                            textual_learner.static_text_features,
                            adaptive_text_features,
                            updated_text_features,
                            delta_normal,
                            delta_abnormal,
                            fixed_global_logit,
                            adaptive_global_logit,
                            final_global_logit,
                            fixed_local_map,
                            adaptive_local_map,
                            final_local_map,
                            total_loss,
                            global_loss,
                            local_loss,
                            anchor_reg_loss,
                            delta_reg_loss,
                            consistency_loss,
                        )
                        if args.use_ranking_loss:
                            logger.info(
                                "Stage II debug: ranking_loss={:.6f}, ranking_weight={:.6f}, "
                                "ranking_score_source={}".format(
                                    ranking_loss.detach().float().item(),
                                    args.ranking_weight,
                                    args.ranking_score_source,
                                )
                            )
                        if args.use_hard_normal_topk_loss:
                            logger.info(
                                "Stage II debug: hard_normal_loss={:.6f}, hard_normal_weight={:.6f}, "
                                "hard_normal_source={}".format(
                                    hard_normal_loss.detach().float().item(),
                                    args.hard_normal_weight,
                                    args.hard_normal_source,
                                )
                            )
                        if args.use_patch_alignment_loss:
                            logger.info(
                                "Stage II debug: patch_alignment_loss={:.6f}, patch_alignment_weight={:.6f}, "
                                "patch_alignment_source={}".format(
                                    patch_alignment_loss.detach().float().item(),
                                    args.patch_alignment_weight,
                                    args.patch_alignment_source,
                                )
                            )
                        if args.use_scr_loss:
                            logger.info(
                                "Stage II debug: scr_loss={:.6f}, scr_loss_weight={:.6f}, "
                                "scr_loss_source={}, scr_margin={:.6f}".format(
                                    scr_loss.detach().float().item(),
                                    args.scr_loss_weight,
                                    args.scr_loss_source,
                                    args.scr_margin,
                                )
                            )
                        if source_memory_stats is not None:
                            logger.info(
                                "Stage II debug: source_memory_train stats={}, weight={:.6f}, "
                                "temperature={:.6f}, topk={}".format(
                                    source_memory_stats,
                                    args.source_memory_weight,
                                    args.source_memory_temperature,
                                    args.source_memory_topk,
                                )
                            )
            else:
                total_loss = local_loss + global_loss
            if stage1_fixed_anchor_visual and len(local_loss_list) == 0:
                log_stage1_batch_debug(logger, global_logit, local_score, total_loss, global_loss, local_loss)
            total_loss.backward()
            if stage2_visual_anchor_update and len(local_loss_list) == 0:
                if reaxis_stage_ii:
                    logger.info("ReAxis Stage II debug: anchor_updater grad_norm: {}".format(module_grad_norm(reaxis_modules.anchor_updater)))
                    logger.info("ReAxis Stage II debug: fusion grad_norm: {}".format(module_grad_norm(reaxis_modules.fusion)))
                    logger.info("ReAxis Stage II debug: image_fusion grad_norm: {}".format(module_grad_norm(reaxis_modules.image_fusion)))
                else:
                    log_stage_ii_grad_debug(logger, anchor_updater, textual_learner, visual_learner)
            optimizer.step()
            total_loss_list.append(total_loss.item())
            global_loss_list.append(global_loss.item())
            local_loss_list.append(local_loss.item())
            if stage2_learnable_anchor:
                anchor_loss_list.append(anchor_reg_loss.item())
            if stage2_visual_anchor_update:
                anchor_loss_list.append(anchor_reg_loss.item())
                delta_loss_list.append(delta_reg_loss.item())
                consistency_loss_list.append(consistency_loss.item())
                if reaxis_stage_ii:
                    reaxis_loss_lists["cond_pixel"].append(cond_pixel_loss.item())
                    reaxis_loss_lists["adapt_pixel"].append(adapt_pixel_loss.item())
                    reaxis_loss_lists["final_pixel"].append(final_pixel_loss.item())
                    reaxis_loss_lists["final_global"].append(final_global_loss.item())
                    reaxis_loss_lists["final_image"].append(final_image_loss.item())
                    reaxis_loss_lists["rotation"].append(rotation_loss.item())
                    reaxis_loss_lists["update"].append(update_loss.item())
                    reaxis_loss_lists["gate"].append(gate_loss.item())
                    reaxis_loss_lists.setdefault("calibration_identity", []).append(calibration_identity_loss.item())
                    reaxis_stat_lists["anchor_update_mean"].append(
                        reaxis_outputs["anchor_update_norm"].detach().float().mean().item()
                    )
                    reaxis_stat_lists["anchor_update_max"].append(
                        reaxis_outputs["anchor_update_norm"].detach().float().max().item()
                    )
                    reaxis_stat_lists["anchor_rotation_mean"].append(
                        reaxis_outputs["anchor_rotation_angle"].detach().float().mean().item()
                    )
                    reaxis_stat_lists["anchor_rotation_max"].append(
                        reaxis_outputs["anchor_rotation_angle"].detach().float().max().item()
                    )
                    reaxis_stat_lists["anchor_rotation_rad_mean"].append(
                        reaxis_outputs["anchor_rotation_angle_rad"].detach().float().mean().item()
                    )
                    reaxis_stat_lists["anchor_rotation_rad_max"].append(
                        reaxis_outputs["anchor_rotation_angle_rad"].detach().float().max().item()
                    )
                if args.use_ranking_loss:
                    ranking_loss_list.append(ranking_loss.item())
                if args.use_hard_normal_topk_loss:
                    hard_normal_loss_list.append(hard_normal_loss.item())
                if args.use_patch_alignment_loss:
                    patch_alignment_loss_list.append(patch_alignment_loss.item())
                if args.use_scr_loss:
                    scr_loss_list.append(scr_loss.item())
                if (
                    not reaxis_stage_ii
                    and args.use_source_memory_residual
                    and getattr(args, "source_memory_train_mode", "none") == "batch"
                    and "source_memory_stats" in locals()
                    and source_memory_stats is not None
                ):
                    source_memory_mean_list.append(source_memory_stats["memory_mean"])

        # logs
        if (epoch + 1) % args.print_freq == 0:
            if stage2_learnable_anchor:
                msg = 'epoch [{}/{}], global_loss:{:.4f}, local_loss:{:.4f}, anchor_reg_loss:{:.4f}'.format(
                    epoch + 1,
                    args.epoch,
                    np.mean(global_loss_list),
                    np.mean(local_loss_list),
                    np.mean(anchor_loss_list),
                )
                if args.use_scr_loss:
                    msg += ', scr_loss:{:.4f}'.format(np.mean(scr_loss_list))
                logger.info(msg)
            elif stage2_visual_anchor_update:
                if reaxis_stage_ii:
                    gates = {
                        key: float(value.detach().cpu().item())
                        for key, value in reaxis_modules.scalar_gates_dict().items()
                    }
                    calibration = reaxis_modules.calibration_parameters_dict()
                    msg = (
                        'epoch [{}/{}], phase:{}, total_loss:{:.4f}, cond_pixel:{:.4f}, '
                        'adapt_pixel:{:.4f}, final_pixel:{:.4f}, final_global:{:.4f}, '
                        'final_image:{:.4f}, rotation:{:.4f}, update:{:.4f}, gate_reg:{:.4f}, cal_identity:{:.4f}, '
                        'update_norm_mean:{:.4f}, update_norm_max:{:.4f}, rotation_angle_deg_mean:{:.4f}, '
                        'rotation_angle_deg_max:{:.4f}, rotation_angle_rad_mean:{:.4f}, rotation_angle_rad_max:{:.4f}, '
                        'gates:{}, calibration:{}'.format(
                            epoch + 1,
                            args.epoch,
                            current_reaxis_phase,
                            np.mean(total_loss_list),
                            np.mean(reaxis_loss_lists["cond_pixel"]),
                            np.mean(reaxis_loss_lists["adapt_pixel"]),
                            np.mean(reaxis_loss_lists["final_pixel"]),
                            np.mean(reaxis_loss_lists["final_global"]),
                            np.mean(reaxis_loss_lists["final_image"]),
                            np.mean(reaxis_loss_lists["rotation"]),
                            np.mean(reaxis_loss_lists["update"]),
                            np.mean(reaxis_loss_lists["gate"]),
                            np.mean(reaxis_loss_lists["calibration_identity"]),
                            np.mean(reaxis_stat_lists["anchor_update_mean"]),
                            np.max(reaxis_stat_lists["anchor_update_max"]),
                            np.mean(reaxis_stat_lists["anchor_rotation_mean"]),
                            np.max(reaxis_stat_lists["anchor_rotation_max"]),
                            np.mean(reaxis_stat_lists["anchor_rotation_rad_mean"]),
                            np.max(reaxis_stat_lists["anchor_rotation_rad_max"]),
                            gates,
                            calibration,
                        )
                    )
                    logger.info(msg)
                else:
                    msg = (
                        'epoch [{}/{}], global_loss:{:.4f}, local_loss:{:.4f}, anchor_reg_loss:{:.4f}, '
                        'delta_reg_loss:{:.4f}, consistency_loss:{:.4f}'.format(
                            epoch + 1,
                            args.epoch,
                            np.mean(global_loss_list),
                            np.mean(local_loss_list),
                            np.mean(anchor_loss_list),
                            np.mean(delta_loss_list),
                            np.mean(consistency_loss_list),
                        )
                    )
                    if args.use_ranking_loss:
                        msg += ', ranking_loss:{:.4f}'.format(np.mean(ranking_loss_list))
                    if args.use_hard_normal_topk_loss:
                        msg += ', hard_normal_loss:{:.4f}'.format(np.mean(hard_normal_loss_list))
                    if args.use_patch_alignment_loss:
                        msg += ', patch_alignment_loss:{:.4f}'.format(np.mean(patch_alignment_loss_list))
                    if args.use_scr_loss:
                        msg += ', scr_loss:{:.4f}'.format(np.mean(scr_loss_list))
                    if source_memory_mean_list:
                        msg += ', source_memory_mean:{:.4f}'.format(np.mean(source_memory_mean_list))
                    logger.info(msg)
            else:
                logger.info('epoch [{}/{}], global_loss:{:.4f}, local_loss:{:.4f}'.format(epoch + 1, args.epoch, np.mean(global_loss_list), np.mean(local_loss_list)))

        # save model
        if (epoch + 1) % args.save_freq == 0:
            ckp_path = os.path.join(args.save_path, 'epoch_' + str(epoch + 1) + '.pth')
            checkpoint = {
                "textual_learner": textual_learner.state_dict(),
                "visual_learner": visual_learner.state_dict(),
                "pq_learner": pq_learner.state_dict(),
                "config": vars(args),
                "epoch": epoch + 1,
                LINEAGE_KEY: build_checkpoint_lineage(
                    args,
                    args.train_stage,
                    parent_checkpoint_path=(
                        args.stage1_checkpoint_path
                        if stage2_learnable_anchor
                        else args.stage2_checkpoint_path if stage2_visual_anchor_update else None
                    ),
                ),
            }
            if stage2_anchor_stage:
                with torch.no_grad():
                    save_prompts, save_tokenized_prompts = textual_learner()
                    checkpoint["base_anchors"] = F.normalize(
                        model.encode_text(save_prompts, save_tokenized_prompts).float(),
                        dim=-1,
                    ).detach().cpu()
            if stage2_visual_anchor_update:
                if reaxis_stage_ii:
                    checkpoint[REAXIS_MODULES_KEY] = reaxis_modules.state_dict()
                    checkpoint[REAXIS_PHASE_KEY] = current_reaxis_phase
                    checkpoint["optimizer"] = optimizer.state_dict()
                else:
                    checkpoint["anchor_updater"] = anchor_updater.state_dict()
            torch.save(checkpoint, ckp_path)



# Compatibility exports for existing preflight tools and scripts.
load_starclip_modules_from_checkpoint = load_reaxis_modules_from_checkpoint
set_starclip_train_phase = set_reaxis_train_phase
starclip_trainable_parameter_groups = reaxis_trainable_parameter_groups
starclip_stage_ii_forward = reaxis_stage_ii_forward

if __name__ == '__main__':
    parser = argparse.ArgumentParser("ReAxis", add_help=True)
    parser.add_argument("--train_data_path", type=str, default="./data/mvtec", help="train dataset path")
    parser.add_argument("--save_path", type=str, default='./checkpoint', help='path to save results')
    parser.add_argument("--dataset", type=str, default='mvtec', help="train dataset name")
    parser.add_argument("--source_validation_mode", type=str, default="none",
                        choices=["none", "category_holdout", "stratified_category_holdout", "image_split"],
                        help="legal source-validation split mode; never uses target test labels")
    parser.add_argument("--source_validation_ratio", type=float, default=0.2,
                        help="source-validation holdout ratio")
    parser.add_argument("--source_validation_seed", type=int, default=111,
                        help="source-validation split seed")
    parser.add_argument("--source_validation_split_path", type=str, default=None,
                        help="existing source-validation split file to reuse")
    parser.add_argument("--source_validation_val_classes", type=str, default=None,
                        help="comma-separated explicit source-validation classes; saved split is then fixed")
    parser.add_argument("--allow_checkpoint_lineage_mismatch", type=str2bool, nargs="?", const=True, default=False,
                        help="debug only: bypass source split/dataset/full-source lineage mismatch errors")
    parser.add_argument("--max_train_batches", type=int, default=None,
                        help="debug/bootstrap only: stop each epoch after this many real dataloader batches")
    parser.add_argument("--pretrained_model", type=str, default='ViT-L/14@336px', help="pre-trained model name")
    parser.add_argument("--n_ctx", type=int, default=12, help="the textual prompt length of textual learner")
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24], help="features used")
    parser.add_argument("--epoch", type=int, default=15, help="epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--image_size", type=int, default=518, help="image size")
    parser.add_argument("--print_freq", type=int, default=1, help="print frequency")
    parser.add_argument("--save_freq", type=int, default=1, help="save frequency")
    parser.add_argument("--seed", type=int, default=10, help="random seed")
    parser.add_argument("--k_shots", type=int, default=1, help="how many normal samples")
    parser.add_argument("--train_stage", type=str, default="original",
                        choices=[
                            "original",
                            "stage_i_source_supervised_base_evidence_learning",
                            "stage_ii_normality_guided_axis_reorientation",
                            "stage_ii_normality_guided_axis_calibration",
                            "stage_i_dual_anchor",
                            "stage_i_dual_anchor_boundary",
                            "stage_ii_visual_evidence",
                            "stage_ii_dual_anchor_visual",
                            "stage1_fixed_anchor_visual",
                            "stage2_learnable_anchor",
                            "stage2_visual_anchor_update",
                        ],
                        help="training stage")
    parser.add_argument("--stage1_checkpoint_path", type=str, default=None,
                        help="pre-existing visual-adapter checkpoint used to initialize ReAxis Stage I")
    parser.add_argument("--stage2_checkpoint_path", type=str, default=None,
                        help="Stage I checkpoint path used to initialize Stage II")
    parser.add_argument("--resume_stage_ii", "--resume_stage22", dest="resume_stage_ii",
                        type=str2bool, nargs="?", const=True, default=False,
                        help="treat stage2_checkpoint_path as a Stage II checkpoint and resume epoch/phase when possible")
    parser.add_argument("--reaxis_preset", "--starclip_preset", "--hprf_preset", dest="starclip_preset", type=str, default="reaxis_full",
                        choices=[
                            "legacy_dual_anchor",
                            "base_dual_anchor_only",
                            "conditioned_dual_anchor_local_only",
                            "reaxis_full",
                            "reaxis_no_calibration",
                            "reaxis_conditioned_global_ablation",
                            "starclip_no_calibration",
                            "starclip_full",
                            "starclip_conditioned_global_ablation",
                            "legacy_stage22",
                            "base_anchor_only",
                            "conditioned_local_only",
                            "hprf_no_calibration",
                            "hprf_full",
                            "hprf_conditioned_global_ablation",
                        ],
                        help="ReAxis Stage II preset/ablation configuration")
    parser.add_argument("--fusion_type", type=str, default="hierarchical_residual",
                        choices=[
                            "hierarchical_residual",
                            "legacy_linear",
                            "base_anchor_only",
                            "conditioned_local_only",
                            "average_mean",
                            "harmonic_mean",
                        ],
                        help="Stage II fusion type; average/harmonic retained for old non-stage2 flows")
    parser.add_argument("--fixed_adaptive_fuse_weight", type=float, default=0.5,
                        help="fixed-anchor branch weight for Stage I/II fusion")
    parser.add_argument("--dual_anchor_margin_fusion_mode", "--legacy_stage22_fusion_mode",
                        dest="dual_anchor_margin_fusion_mode", type=str, default="legacy_linear",
                        choices=[
                            "legacy_linear",
                            "global_base_local_conditioned",
                            "margin_fixed_conditioned",
                            "margin_global_base_local_conditioned",
                        ],
                        help="optional low-risk legacy Stage II fusion ablation")
    parser.add_argument("--legacy_alpha_global", type=float, default=None,
                        help="fixed/global weight for legacy Stage II fusion ablations; defaults to fixed_adaptive_fuse_weight")
    parser.add_argument("--legacy_alpha_local", type=float, default=None,
                        help="fixed/local weight for legacy Stage II fusion ablations; defaults to fixed_adaptive_fuse_weight")
    parser.add_argument("--patch_ms_agg", type=str2bool, nargs="?", const=True, default=False,
                        help="enable ReAxis multi-scale Gaussian patch-token aggregation before local anchor matching")
    parser.add_argument("--patch_ms_kernel_sizes", type=int, nargs="+", default=[1, 3, 5],
                        help="odd kernel sizes for multi-scale patch-token aggregation")
    parser.add_argument("--patch_ms_sigma", type=float, default=4.0,
                        help="Gaussian sigma for multi-scale patch-token aggregation")
    parser.add_argument("--patch_ms_fuse", type=str, default="residual", choices=["mean", "residual"],
                        help="fusion rule for multi-scale patch-token aggregation")
    parser.add_argument("--patch_ms_residual_beta", type=float, default=0.1,
                        help="residual beta when patch_ms_fuse=residual")
    parser.add_argument("--patch_ms_apply_to", type=str, default="none",
                        choices=["none", "base", "conditioned", "base_and_conditioned", "all_textual"],
                        help="which textual-anchor local branches use multi-scale patch-token aggregation")
    parser.add_argument("--use_source_memory_residual", type=str2bool, nargs="?", const=True, default=False,
                        help="enable source-only CLIP patch memory residual during Stage II training")
    parser.add_argument("--source_memory_train_mode", type=str, default="none", choices=["none", "batch"],
                        help="source-memory construction mode for training; batch uses the current labeled source batch")
    parser.add_argument("--source_memory_max_normal_patches", type=int, default=4096,
                        help="maximum normal source patches used by training batch memory")
    parser.add_argument("--source_memory_max_anomaly_patches", type=int, default=4096,
                        help="maximum anomaly source patches used by training batch memory")
    parser.add_argument("--source_memory_weight", type=float, default=0.1,
                        help="blend weight for source-memory local probability residual")
    parser.add_argument("--source_memory_temperature", type=float, default=10.0,
                        help="temperature applied to abnormal-vs-normal source memory evidence")
    parser.add_argument("--source_memory_topk", type=int, default=5,
                        help="top-k source memory neighbors averaged for normal/anomaly evidence")
    parser.add_argument("--source_memory_chunk_size", type=int, default=4096,
                        help="target patch chunk size for source memory similarity computation")
    parser.add_argument("--use_patch_alignment_loss", type=str2bool, nargs="?", const=True, default=False,
                        help="enable PAL-style patch score alignment loss for legacy Stage II")
    parser.add_argument("--patch_alignment_weight", type=float, default=0.0,
                        help="weight for patch score alignment loss")
    parser.add_argument("--patch_alignment_source", type=str, default="adaptive",
                        choices=["adaptive", "final", "both"],
                        help="local score source for patch score alignment")
    parser.add_argument("--use_fb_suppression", type=str2bool, nargs="?", const=True, default=False,
                        help="enable fixed-prior foreground/background suppression for textual local branches")
    parser.add_argument("--fb_suppression_apply_to", type=str, default="none",
                        choices=["none", "base", "conditioned", "base_and_conditioned", "all_textual", "final"],
                        help="which local branch receives foreground/background suppression")
    parser.add_argument("--fb_suppression_strength", type=float, default=0.15,
                        help="maximum abnormal-probability attenuation in predicted background")
    parser.add_argument("--fb_suppression_threshold", type=float, default=0.5,
                        help="fixed-prior abnormal probability threshold for foreground confidence")
    parser.add_argument("--fb_suppression_temperature", type=float, default=10.0,
                        help="sigmoid temperature for foreground confidence")
    parser.add_argument("--use_scr_loss", type=str2bool, nargs="?", const=True, default=False,
                        help="enable source-mask semantic consistency regularization")
    parser.add_argument("--scr_loss_weight", type=float, default=0.0,
                        help="weight for semantic consistency regularization")
    parser.add_argument("--scr_loss_source", type=str, default="adaptive",
                        choices=["adaptive", "final", "both", "base"],
                        help="local score source for semantic consistency regularization")
    parser.add_argument("--scr_margin", type=float, default=0.5,
                        help="target abnormal-normal evidence margin for SCR")
    parser.add_argument("--anchor_reg_weight", type=float, default=0.01,
                        help="weight for Stage I/II anchor regularization")
    parser.add_argument("--freeze_visual_adapter_in_stage2", type=str2bool, nargs="?", const=True, default=True,
                        help="freeze VisualAdapter during Stage I")
    parser.add_argument("--anchor_update_gamma", type=float, default=0.05,
                        help="Stage II anchor residual update scale")
    parser.add_argument("--anchor_update_hidden_dim", type=int, default=512,
                        help="Stage II anchor updater hidden dimension")
    parser.add_argument("--anchor_update_mode", type=str, default="contrast_direction",
                        choices=["contrast_direction", "legacy_independent"],
                        help="Stage II anchor update mode")
    parser.add_argument("--anchor_update_gamma_max", type=float, default=0.1,
                        help="legacy alias for ReAxis anchor tangent cap; prefer --anchor_update_max_angle_deg")
    parser.add_argument("--anchor_update_max_angle_deg", type=float, default=5.0,
                        help="strict maximum ReAxis anchor contrast rotation angle in degrees")
    parser.add_argument("--learnable_anchor_update_gamma", type=str2bool, nargs="?", const=True, default=False,
                        help="use sigmoid-bounded learnable gamma in [0, gamma_max]")
    parser.add_argument("--freeze_base_anchors_in_stage_ii", "--freeze_base_anchors_in_stage22",
                        dest="freeze_base_anchors_in_stage_ii", type=str2bool, nargs="?", const=True, default=True,
                        help="freeze Stage I base anchors during Stage II")
    parser.add_argument("--zero_init_anchor_updater", type=str2bool, nargs="?", const=True, default=True,
                        help="zero initialize ReAxis updater final layer")
    parser.add_argument("--conditioned_anchor_scope", type=str, default="local",
                        choices=["local", "global", "global_and_local"],
                        help="where image-conditioned anchors are allowed to contribute")
    parser.add_argument("--anchor_condition_source", type=str, default="global_plus_normal_context",
                        choices=["global_only", "normal_context_only", "global_plus_normal_context"],
                        help="feature source for ReAxis anchor conditioning")
    parser.add_argument("--normal_context_beta", type=float, default=10.0,
                        help="softmax concentration for fixed-map normal context extraction")
    parser.add_argument("--detach_normal_context_weights", type=str2bool, nargs="?", const=True, default=True,
                        help="detach fixed anomaly map before normal-context weighting")
    parser.add_argument("--enable_branch_calibration", type=str2bool, nargs="?", const=True, default=True,
                        help="enable ScalarAffineCalibrator per branch/task")
    parser.add_argument("--calibration_temperature_min", type=float, default=0.05,
                        help="minimum scalar calibrator temperature")
    parser.add_argument("--calibration_temperature_max", type=float, default=20.0,
                        help="maximum scalar calibrator temperature")
    parser.add_argument("--freeze_calibration_during_updater_warmup", type=str2bool, nargs="?", const=True, default=True,
                        help="do not train calibrators in Stage II updater warm-up")
    parser.add_argument("--condition_gate_mode", type=str, default="scalar", choices=["scalar"],
                        help="ReAxis conditioned-local gate mode")
    parser.add_argument("--adapt_global_gate_mode", type=str, default="scalar", choices=["scalar"],
                        help="ReAxis adaptive-global gate mode")
    parser.add_argument("--adapt_local_gate_mode", type=str, default="scalar", choices=["scalar"],
                        help="ReAxis adaptive-local gate mode")
    parser.add_argument("--condition_gate_init", type=float, default=0.15,
                        help="initial conditioned-local residual gate")
    parser.add_argument("--adapt_global_gate_init", type=float, default=0.4,
                        help="initial Stage II global residual gate")
    parser.add_argument("--adapt_local_gate_init", type=float, default=0.4,
                        help="initial Stage II local residual gate")
    parser.add_argument("--condition_residual_cap", type=float, default=4.0,
                        help="conditioned-local bounded residual cap in evidence space")
    parser.add_argument("--global_residual_cap", type=float, default=4.0,
                        help="global bounded residual cap in evidence space")
    parser.add_argument("--local_residual_cap", type=float, default=4.0,
                        help="local bounded residual cap in evidence space")
    parser.add_argument("--image_local_pooling", type=str, default="topk_mean",
                        choices=["topk_mean", "max", "logsumexp"],
                        help="pooling for raw final local probability to image-level evidence")
    parser.add_argument("--image_local_pooling_space", type=str, default="native_grid",
                        choices=["native_grid", "final_map"],
                        help="space used for image-level local pooling before Gaussian smoothing")
    parser.add_argument("--image_local_topk_ratio", type=float, default=0.01,
                        help="Top-K ratio for image-level local pooling")
    parser.add_argument("--gaussian_for_image_score", type=str2bool, nargs="?", const=True, default=False,
                        help="legacy ablation: allow smoothed map to affect image score")
    parser.add_argument("--image_score_fusion", type=str, default="calibrated_logit_mix",
                        choices=["calibrated_logit_mix", "legacy_arithmetic_average"],
                        help="image-level score fusion")
    parser.add_argument("--image_gate_init", type=float, default=0.5,
                        help="initial global/local image evidence gate")
    parser.add_argument("--sigma", type=float, default=4.0,
                        help="Gaussian sigma for eval/visual anomaly map only")
    parser.add_argument("--loss_cond_pixel_weight", type=float, default=1.0,
                        help="Stage II conditioned-local standalone pixel loss weight")
    parser.add_argument("--loss_adapt_pixel_weight", type=float, default=0.5,
                        help="Stage II adaptive-local hierarchical pixel loss weight")
    parser.add_argument("--loss_final_pixel_weight", type=float, default=1.0,
                        help="Stage II final fused local pixel loss weight")
    parser.add_argument("--loss_final_global_weight", type=float, default=1.0,
                        help="Stage II final global BCE loss weight")
    parser.add_argument("--loss_final_image_weight", type=float, default=0.5,
                        help="Stage II final image-score BCE loss weight")
    parser.add_argument("--loss_anchor_rotation_weight", type=float, default=0.05,
                        help="Stage II anchor rotation regularization weight")
    parser.add_argument("--loss_anchor_update_weight", type=float, default=0.01,
                        help="Stage II updater residual norm regularization weight")
    parser.add_argument("--loss_calibration_identity_weight", type=float, default=1e-3,
                        help="weight for scalar calibrator identity regularization")
    parser.add_argument("--calibration_bias_regularization_ratio", type=float, default=1.0,
                        help="relative bias penalty inside calibration identity regularization")
    parser.add_argument("--loss_gate_regularization_weight", type=float, default=1e-4,
                        help="Stage II conservative gate regularization weight")
    parser.add_argument("--gate_regularization_mode", type=str, default="initial",
                        choices=["initial", "zero"],
                        help="regularize gates to their initial values or legacy zero target")
    parser.add_argument("--stage_ii_updater_warmup_epochs", "--stage22_updater_warmup_epochs",
                        dest="stage_ii_updater_warmup_epochs", type=int, default=None,
                        help="explicit epochs that train updater before fusion/calibration")
    parser.add_argument("--stage_ii_updater_warmup_ratio", "--stage22_updater_warmup_ratio",
                        dest="stage_ii_updater_warmup_ratio", type=float, default=0.2,
                        help="warm-up ratio used when explicit warm-up epochs is omitted")
    parser.add_argument("--updater_lr", type=float, default=None,
                        help="ReAxis updater learning rate")
    parser.add_argument("--fusion_lr", type=float, default=None,
                        help="ReAxis gate/image fusion learning rate")
    parser.add_argument("--calibration_lr", type=float, default=None,
                        help="ReAxis branch calibration learning rate")
    parser.add_argument("--delta_reg_weight", type=float, default=0.001,
                        help="weight for Stage II delta norm regularization")
    parser.add_argument("--anchor_update_cons_weight", type=float, default=0.05,
                        help="weight for Stage II updated adaptive/fixed map consistency")
    parser.add_argument("--use_ranking_loss", type=str2bool, nargs="?", const=True, default=False,
                        help="enable Stage II pairwise ranking loss for image-level AUROC")
    parser.add_argument("--ranking_weight", type=float, default=0.1,
                        help="weight for Stage II pairwise ranking loss")
    parser.add_argument("--ranking_margin", type=float, default=0.1,
                        help="margin for Stage II pairwise ranking loss")
    parser.add_argument("--ranking_score_source", type=str, default="global",
                        choices=["global", "global_plus_topk", "map_max"],
                        help="score source for Stage II pairwise ranking loss")
    parser.add_argument("--ranking_topk_ratio", type=float, default=0.001,
                        help="top-k ratio used when ranking_score_source=global_plus_topk")
    parser.add_argument("--ranking_max_pairs", type=int, default=0,
                        help="maximum hard pairs for ranking loss; <=0 uses all pairs")
    parser.add_argument("--use_hard_normal_topk_loss", type=str2bool, nargs="?", const=True, default=False,
                        help="enable Stage II hard normal top-k pixel suppression loss")
    parser.add_argument("--hard_normal_weight", type=float, default=0.05,
                        help="weight for Stage II hard normal top-k pixel suppression loss")
    parser.add_argument("--hard_normal_topk_ratio", type=float, default=0.001,
                        help="top-k pixel ratio for hard normal suppression")
    parser.add_argument("--hard_normal_margin", type=float, default=0.0,
                        help="allowed normal-image top-k anomaly score before suppression")
    parser.add_argument("--hard_normal_source", type=str, default="final",
                        choices=["final", "updated", "fixed"],
                        help="map source for hard normal top-k suppression")
    parser.add_argument("--freeze_textual_adapter_in_stage_ii", "--freeze_textual_adapter_in_stage22",
                        dest="freeze_textual_adapter_in_stage_ii", type=str2bool, nargs="?", const=True, default=True,
                        help="freeze TextualAdapter during Stage II")
    parser.add_argument("--freeze_visual_adapter_in_stage_ii", "--freeze_visual_adapter_in_stage22",
                        dest="freeze_visual_adapter_in_stage_ii", type=str2bool, nargs="?", const=True, default=True,
                        help="freeze VisualAdapter during Stage II")
    parser.add_argument("--use_visual_lsar", type=str2bool, nargs="?", const=True, default=False,
                        help="enable LSAR-lite visual residual adapters inside VisualAdapter")
    parser.add_argument("--visual_lsar_bottleneck_ratio", type=int, default=4,
                        help="bottleneck ratio for LSAR-lite visual residual adapters")
    parser.add_argument("--visual_lsar_scale", type=float, default=1.0,
                        help="residual scale for LSAR-lite visual residual adapters")
    parser.add_argument("--train_visual_lsar_when_frozen", type=str2bool, nargs="?", const=True, default=True,
                        help="train LSAR-lite residual parameters when VisualAdapter is otherwise frozen")
    parser.add_argument("--visual_learner", action="store_true", help="Enable visual adapter")
    parser.add_argument("--textual_learner", action="store_true", help="Enable textual adapter")
    parser.add_argument("--pq_learner", action="store_true", help="Enable prompt-query adapter")
    parser.add_argument("--vl_reduction", type=int, default=4, help="the reduction number of visual learner")
    parser.add_argument("--pq_mid_dim", type=int, default=128, help="the number of the first hidden layer in pqadapter")
    parser.add_argument("--pq_context", action="store_true", help="Enable context feature")

    args = parser.parse_args()
    setup_seed(args.seed)
    train(args)
