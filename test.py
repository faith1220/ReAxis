# ReAxis modifications, 2026-09-09: public naming, compatibility and release packaging.
"""Testing script for the ReAxis anomaly detection model."""

import argparse
import csv
import os
import pickle
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from tabulate import tabulate
from tqdm import tqdm

import reaxislib
from adaptcliplib.reaxis import LEGACY_HPRF_MODULES_KEY
from reaxislib import (PQAdapter, TextualAdapter, VisionConditionedAnchorUpdater,
                          REAXIS_MODULES_KEY,
                          ReAxisStageIIModules, VisualAdapter,
                          apply_source_memory_residual,
                          apply_reaxis_preset, binary_margin,
                          build_query_derived_normal_context,
                          compute_global_local_score_batchwise,
                          foreground_background_suppression,
                          gaussian_smoothing_2d, legacy_linear_fusion,
                          dual_anchor_margin_fusion,
                          probability_to_margin, topk_mean, uses_reaxis,
                          source_memory_anomaly_probability,
                          source_patch_memory_from_features,
                          fusion_fun)
from dataset import Dataset, PromptDataset
from tools import Evaluator, get_logger, get_transform, setup_seed, stable_source_split_hash, visualizer


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


def to_scalar_string(value):
    if torch.is_tensor(value):
        value = value.detach().cpu()
        if value.ndim == 0:
            return str(value.item())
        return str(value.tolist())
    return str(value)


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


def load_reaxis_modules_from_checkpoint(reaxis_modules, checkpoint, logger, source):
    if reaxis_modules is None:
        return
    module_key = REAXIS_MODULES_KEY if REAXIS_MODULES_KEY in checkpoint else LEGACY_HPRF_MODULES_KEY
    if module_key not in checkpoint:
        logger.info(
            "ReAxis modules not found in {}; using initialized structured updater, "
            "identity calibration, and configured gates".format(source)
        )
        return
    incompatible = reaxis_modules.load_state_dict(checkpoint[module_key], strict=False)
    if incompatible.missing_keys:
        logger.info("ReAxis missing keys kept at initialization: {}".format(incompatible.missing_keys))
    if incompatible.unexpected_keys:
        logger.info("ReAxis unexpected keys ignored with strong warning: {}".format(incompatible.unexpected_keys))


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


def branch_image_score(reaxis_modules, global_evidence, local_evidence, sigma, gaussian_for_image_score=False, pooling_size=None):
    out = reaxis_modules.image_fusion(
        global_evidence,
        local_evidence,
        gaussian_sigma=sigma,
        gaussian_for_image_score=gaussian_for_image_score,
        pooling_size=pooling_size,
    )
    return out["image_score"], out["pixel_prob_eval"]


def legacy_local_image_score(raw_pixel_map, smoothed_pixel_map, mode, topk_ratio):
    """Pool local anomaly evidence for legacy image-level scoring."""
    if mode in ("legacy_smoothed_max", "smoothed_max"):
        return smoothed_pixel_map.reshape(smoothed_pixel_map.shape[0], -1).max(dim=1).values
    if mode == "raw_max":
        return raw_pixel_map.reshape(raw_pixel_map.shape[0], -1).max(dim=1).values
    if mode == "raw_topk_mean":
        return topk_mean(raw_pixel_map, topk_ratio)
    if mode == "smoothed_topk_mean":
        return topk_mean(smoothed_pixel_map, topk_ratio)
    raise ValueError("unsupported legacy_image_score_local_pooling: {}".format(mode))


def _cap_memory_bank(features, max_count):
    if features.numel() == 0 or max_count is None or max_count <= 0 or features.shape[0] <= max_count:
        return features
    idx = torch.linspace(0, features.shape[0] - 1, steps=int(max_count), device=features.device).long()
    return features.index_select(0, idx)


def _append_capped_memory_bank(chunks, new_chunk, max_count):
    if new_chunk.numel() == 0:
        return chunks
    chunks.append(new_chunk.detach().float().cpu())
    merged = torch.cat(chunks, dim=0)
    merged = _cap_memory_bank(merged, max_count)
    return [merged]


def build_source_memory_bank(args, model, preprocess, target_transform, device, logger, DPAM_layer):
    if not getattr(args, "use_source_memory_residual", False):
        return None
    if not args.source_memory_data_path or not args.source_memory_dataset:
        raise ValueError("--source_memory_data_path and --source_memory_dataset are required when source memory is enabled")

    source_data = Dataset(
        root=args.source_memory_data_path,
        transform=preprocess,
        target_transform=target_transform,
        dataset_name=args.source_memory_dataset,
        k_shots=args.k_shots,
        save_dir=args.save_path,
        mode="test",
        seed=args.seed,
    )
    source_loader = torch.utils.data.DataLoader(
        source_data,
        batch_size=args.source_memory_batch_size,
        shuffle=False,
        num_workers=4,
    )
    normal_bank = []
    anomaly_bank = []
    logger.info(
        "building source patch memory: dataset={}, path={}, samples={}".format(
            args.source_memory_dataset,
            args.source_memory_data_path,
            len(source_data),
        )
    )
    for items in tqdm(source_loader, desc="source_memory"):
        image = items["img"].to(device)
        mask = items["img_mask"][:, 0].to(device)
        with torch.no_grad():
            _, patch_feats = model.encode_image(image, args.features_list, DPAM_layer=DPAM_layer)
        normal, anomaly = source_patch_memory_from_features(
            patch_feats[-1],
            mask,
            max_normal_patches=args.source_memory_max_normal_patches,
            max_anomaly_patches=args.source_memory_max_anomaly_patches,
        )
        if normal.numel() > 0:
            normal_bank = _append_capped_memory_bank(
                normal_bank,
                normal,
                args.source_memory_max_normal_patches,
            )
        if anomaly.numel() > 0:
            anomaly_bank = _append_capped_memory_bank(
                anomaly_bank,
                anomaly,
                args.source_memory_max_anomaly_patches,
            )

    normal_bank = torch.cat(normal_bank, dim=0) if normal_bank else torch.empty(0, 0)
    anomaly_bank = torch.cat(anomaly_bank, dim=0) if anomaly_bank else torch.empty(0, 0)
    normal_bank = _cap_memory_bank(normal_bank, args.source_memory_max_normal_patches)
    anomaly_bank = _cap_memory_bank(anomaly_bank, args.source_memory_max_anomaly_patches)
    if normal_bank.numel() == 0 or anomaly_bank.numel() == 0:
        raise ValueError(
            "source memory audit failed: normal_patches={}, anomaly_patches={}".format(
                normal_bank.shape[0] if normal_bank.ndim else 0,
                anomaly_bank.shape[0] if anomaly_bank.ndim else 0,
            )
        )
    logger.info(
        "source patch memory ready: normal={}, anomaly={}, dim={}, weight={}, topk={}, temperature={}".format(
            normal_bank.shape[0],
            anomaly_bank.shape[0],
            normal_bank.shape[1],
            args.source_memory_weight,
            args.source_memory_topk,
            args.source_memory_temperature,
        )
    )
    return {
        "normal": normal_bank.contiguous().to(device),
        "anomaly": anomaly_bank.contiguous().to(device),
    }


def prompt_association(image_memory, patch_memory, target_class_name):
    patch_level_num = len(patch_memory[target_class_name[0]])
    retrive_image = []
    retrive_patch = [[] for i in range(patch_level_num)]

    for class_name in target_class_name:
        retrive_image.append(image_memory[class_name])  # S*D
        for l in range(patch_level_num):
            retrive_patch[l].append(patch_memory[class_name][l]) #

    retrive_image = torch.stack(retrive_image)  # B*S*D
    for l in range(patch_level_num):
        retrive_patch[l] = torch.stack(retrive_patch[l])  # B*S*L*D
    return retrive_image, retrive_patch


def build_prompt_memory(model, prompt_dataloader, device, obj_list, view_list, features_list, DPAM_layer):
    """Build few-shot prompt memory."""
    # initialize_memory
    feats_scale_num = len(features_list)
    prompt_image_memory = {}
    prompt_patch_memory = {}

    image_temp = []
    patch_temp = [[] for i in range(feats_scale_num)]
    cls_names_temp = []
    view_ids_temp = []

    for idx, items in enumerate(tqdm(prompt_dataloader)):
        cls_name = items['cls_name']
        prompt_image = items['img'].to(device)  # B*s*c*h*w
        prompt_mask = items['img_mask'].to(device)
        view_id = items['view_id']

        with torch.no_grad():
            image_feat, patch_feat = model.encode_image(prompt_image, features_list, DPAM_layer = DPAM_layer)

        cls_names_temp.extend(cls_name)
        image_temp.append(image_feat)
        view_ids_temp.extend(view_id)

        for i in range(feats_scale_num):
            patch_temp[i].append(patch_feat[i])


    image_temp = torch.cat(image_temp, dim=0)
    for i in range(feats_scale_num):
        patch_temp[i] = torch.cat(patch_temp[i], dim=0)

    for obj in obj_list:
        if len(view_list) > 1:
            for view_id in view_list:
                indice = (np.array(cls_names_temp) == obj) & (np.array(view_ids_temp) == view_id)
                obj_name = obj + '_' + view_id

                prompt_image_memory[obj_name] = image_temp[indice]
                prompt_patch_memory[obj_name] = []

                for i in range(feats_scale_num):
                    prompt_patch_memory[obj_name].append(patch_temp[i][[indice]])
        else:
            indice = (np.array(cls_names_temp) == obj)
            obj_name = obj

            prompt_image_memory[obj_name] = image_temp[indice]
            prompt_patch_memory[obj_name] = []

            for i in range(feats_scale_num):
                prompt_patch_memory[obj_name].append(patch_temp[i][[indice]])

    return prompt_image_memory, prompt_patch_memory


def test(args):
    args.train_stage = normalize_train_stage_name(args.train_stage)
    img_size = args.image_size
    features_list = args.features_list
    dataset_dir = args.test_data_path
    save_path = args.save_path
    dataset_name = args.dataset
    batch_size = args.batch_size
    k_shots = args.k_shots
    seed = args.seed
    vl_reduction = args.vl_reduction
    pq_mid_dim = args.pq_mid_dim
    pq_context = args.pq_context
    eval_metrics =  args.eval_metrics
    stage1_fixed_anchor_visual = args.train_stage == "stage1_fixed_anchor_visual"
    stage2_learnable_anchor = args.train_stage == "stage2_learnable_anchor"
    stage2_visual_anchor_update = args.train_stage == "stage2_visual_anchor_update"
    stage2_anchor_stage = stage2_learnable_anchor or stage2_visual_anchor_update
    adapter_only_stage = stage1_fixed_anchor_visual or stage2_anchor_stage
    if stage2_visual_anchor_update:
        apply_reaxis_preset(args)
    reaxis_stage_ii = uses_reaxis(args)
    mode = 'test'

    log_file = f'{dataset_name}_{seed}seed_{k_shots}shot_{mode}_log.txt'
    logger = get_logger(save_path, log_file)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.pretrained_model == 'ViT-L/14@336px':
        model, _ = reaxislib.load(args.pretrained_model, device=device)
        model.visual.DAPM_replace(DPAM_layer = 20)
        patch_size = 14
        input_dim = 768
        DPAM_layer = 20
    if args.pretrained_model == 'VITB16_PLUS_240':
        model, _ = reaxislib.load(args.pretrained_model, device=device)
        model.visual.DAPM_replace(DPAM_layer = 10)
        patch_size = 16
        input_dim = 640
        DPAM_layer = 10

    preprocess, target_transform = get_transform(image_size=args.image_size)
    if dataset_name in ['Real-IAD-Variety', 'RealIAD']:
        sample_level = True
        prompt_data = PromptDataset(root=dataset_dir, transform=preprocess, target_transform=target_transform, \
                                    dataset_name=dataset_name, k_shots=k_shots, save_dir=save_path, mode=mode, \
                                    seed=seed, class_name=args.class_name)
        test_data = Dataset(root=dataset_dir, transform=preprocess, target_transform=target_transform, \
                            dataset_name=dataset_name, k_shots=k_shots, save_dir=save_path, mode=mode, \
                            seed=seed, class_name=args.class_name,
                            source_split_path=args.source_validation_split_path,
                            source_split_role=args.source_validation_role)
    else:
        prompt_data = PromptDataset(root=dataset_dir, transform=preprocess, target_transform=target_transform, \
                                    dataset_name=dataset_name, k_shots=k_shots, save_dir=save_path, mode=mode, seed=seed)
        test_data = Dataset(root=dataset_dir, transform=preprocess, target_transform=target_transform, \
                            dataset_name=dataset_name, k_shots=k_shots, save_dir=save_path, mode=mode, seed=seed,
                            source_split_path=args.source_validation_split_path,
                            source_split_role=args.source_validation_role)
        sample_level = False
    prompt_dataloader = torch.utils.data.DataLoader(prompt_data, batch_size=batch_size, shuffle=False)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4)
    obj_list = test_data.obj_list
    view_list = test_data.view_list

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


    logger.info('\n' + "loading model from: " + args.checkpoint_path)
    checkpoint_adapter = torch.load(args.checkpoint_path, map_location="cpu")
    if stage1_fixed_anchor_visual:
        if "textual_learner" in checkpoint_adapter:
            textual_learner.load_state_dict(checkpoint_adapter["textual_learner"])
        load_visual_adapter_state(visual_learner, checkpoint_adapter["visual_learner"], logger, args.checkpoint_path)
        if "pq_learner" in checkpoint_adapter:
            pq_learner.load_state_dict(checkpoint_adapter["pq_learner"])
    elif stage2_learnable_anchor:
        missing_keys = [key for key in ["textual_learner", "visual_learner"] if key not in checkpoint_adapter]
        if missing_keys:
            raise KeyError("Stage I checkpoint missing keys: {}".format(missing_keys))
        textual_learner.load_state_dict(checkpoint_adapter["textual_learner"])
        load_visual_adapter_state(visual_learner, checkpoint_adapter["visual_learner"], logger, args.checkpoint_path)
        if "pq_learner" in checkpoint_adapter:
            pq_learner.load_state_dict(checkpoint_adapter["pq_learner"])
    elif stage2_visual_anchor_update:
        expected = ["textual_learner", "visual_learner"] if reaxis_stage_ii else ["textual_learner", "visual_learner", "anchor_updater"]
        missing_keys = [key for key in expected if key not in checkpoint_adapter]
        if missing_keys:
            raise KeyError("Stage II checkpoint missing keys: {}".format(missing_keys))
        textual_learner.load_state_dict(checkpoint_adapter["textual_learner"])
        load_visual_adapter_state(visual_learner, checkpoint_adapter["visual_learner"], logger, args.checkpoint_path)
        if reaxis_stage_ii:
            load_reaxis_modules_from_checkpoint(reaxis_modules, checkpoint_adapter, logger, args.checkpoint_path)
        else:
            anchor_updater.load_state_dict(checkpoint_adapter["anchor_updater"])
        if "pq_learner" in checkpoint_adapter:
            pq_learner.load_state_dict(checkpoint_adapter["pq_learner"])
    else:
        textual_learner.load_state_dict(checkpoint_adapter["textual_learner"])
        load_visual_adapter_state(visual_learner, checkpoint_adapter["visual_learner"], logger, args.checkpoint_path)
        pq_learner.load_state_dict(checkpoint_adapter["pq_learner"])


    model.to(device)
    textual_learner.to(device)
    visual_learner.to(device)
    pq_learner.to(device)
    if anchor_updater is not None:
        anchor_updater.to(device)
    if reaxis_modules is not None:
        reaxis_modules.to(device)

    model.eval()
    textual_learner.eval()
    visual_learner.eval()
    pq_learner.eval()
    if anchor_updater is not None:
        anchor_updater.eval()
    if reaxis_modules is not None:
        reaxis_modules.eval()


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


    # ====================== Initialize Evaluation Metrics ======================
    cpu_eva = args.cpu_eva
    logger.info("evaluation metrics run on CPU; cpu_eva controls prediction storage")
    if args.aupro_num_thresholds is not None:
        logger.info("P-AUPRO will use {} thresholds".format(args.aupro_num_thresholds))
    if cpu_eva:
        evaluator = Evaluator('cpu', metrics=eval_metrics, sample_level=sample_level, aupro_num_thresholds=args.aupro_num_thresholds)
    else:
        evaluator = Evaluator(device, metrics=eval_metrics, sample_level=sample_level, aupro_num_thresholds=args.aupro_num_thresholds)

    source_memory_bank = build_source_memory_bank(args, model, preprocess, target_transform, device, logger, DPAM_layer)

    # ======================Text Encoder forward ======================
    textual_learner.prepare_static_text_feature(model)
    static_text_features = textual_learner.static_text_features.detach() if adapter_only_stage else textual_learner.static_text_features

    if not adapter_only_stage:
        learned_prompts, tokenized_prompts = textual_learner()
        learned_text_features = model.encode_text_learn(learned_prompts, tokenized_prompts).float()
    elif stage2_anchor_stage:
        learned_prompts, tokenized_prompts = textual_learner()
        learned_text_features = model.encode_text(learned_prompts, tokenized_prompts).float()
        learned_text_features = F.normalize(learned_text_features, dim=-1)


    # ====================== Few-shot Prompt Memory ======================
    if k_shots > 0 and not adapter_only_stage:
        prompt_image_memory, prompt_patch_memory = build_prompt_memory(model, prompt_dataloader, device, obj_list, view_list, args.features_list, DPAM_layer)


    # ====================== Visual and Learner forward ======================
    sample_ids, gt_masks, pr_masks, cls_names, gt_anomalys, pr_anomalys, query_paths = [], [], [], [], [], [], []
    stage2_branch_rows = []
    branch_eval_predictions = defaultdict(lambda: {"pr_masks": [], "pr_anomalys": []})
    visual_counts = defaultdict(int)
    visual_saved_total = 0
    # nums = 0
    # total_time = 0
    for idx, items in enumerate(tqdm(test_dataloader)):
        query_image = items['img'].to(device)
        current_batchsize = query_image.shape[0]
        query_path = items['img_path']

        cls_name = items['cls_name']
        cls_id = items['cls_id']
        sample_id = items['sample_id']

        gt_anomaly = items['anomaly'].to(device)
        gt_mask = items['img_mask'][:, 0]
        gt_mask[gt_mask > 0.5], gt_mask[gt_mask <= 0.5] = 1, 0
        gt_mask = gt_mask.to(device)

        # torch.cuda.synchronize()
        # start_time = time.time()

        with torch.no_grad():
            query_feats, query_patch_feats = model.encode_image(query_image, args.features_list, DPAM_layer = DPAM_layer)

        if k_shots > 0 and not adapter_only_stage:
            if len(view_list) > 1:
                target_cls_name = [cls_name + '_' + view_id for cls_name, view_id in zip(cls_name, items['view_id'])]
            else:
                target_cls_name = cls_name
            prompt_feats, prompt_patch_feats = prompt_association(prompt_image_memory, prompt_patch_memory, target_cls_name)

        # ====================== CLIP Baseline ======================
        '''
        global_logit, local_map = textual_learner.compute_global_local_score(query_feats, query_patch_feats, static_text_features)
        local_map = local_map[:, 1].detach()

        global_score = global_logit.softmax(-1)
        global_score = global_score[:, 1].detach()
        '''

        # ====================== visual_adapter ======================
        if args.visual_learner or adapter_only_stage:
            global_vl_logit, local_vl_score = visual_learner(query_feats, query_patch_feats, static_text_features)
            local_vl_map = local_vl_score[:, 1].detach()

            global_vl_score = global_vl_logit.softmax(-1)
            global_vl_score = global_vl_score[:, 1].detach()

        # ====================== textual_adapter ======================
        if args.textual_learner and not adapter_only_stage:
            global_tl_logit, local_tl_map = textual_learner.compute_global_local_score(query_feats, query_patch_feats, learned_text_features)
            local_tl_map = local_tl_map[:, 1].detach()

            global_tl_score = global_tl_logit.softmax(-1)
            global_tl_score = global_tl_score[:, 1].detach()

        if stage2_learnable_anchor:
            adaptive_global_logit, adaptive_local_score = textual_learner.compute_global_local_score(
                query_feats,
                query_patch_feats,
                learned_text_features,
                **patch_ms_kwargs_for_branch(args, "base"),
            )
            adaptive_local_score = maybe_apply_fb_suppression(
                args,
                "base",
                adaptive_local_score,
                local_vl_score.detach(),
            )
        elif stage2_visual_anchor_update:
            base_global_logit = None
            base_local_score = None
            legacy_needs_base = getattr(args, "dual_anchor_margin_fusion_mode", "legacy_linear") in (
                "global_base_local_conditioned",
                "margin_global_base_local_conditioned",
            )
            if reaxis_stage_ii:
                base_global_logit, base_local_score = textual_learner.compute_global_local_score(
                    query_feats,
                    query_patch_feats,
                    learned_text_features,
                    **patch_ms_kwargs_for_branch(args, "base"),
                )
                reaxis_outputs = reaxis_stage_ii_forward(
                    args,
                    reaxis_modules,
                    query_feats,
                    query_patch_feats,
                    global_vl_logit,
                    local_vl_score,
                    learned_text_features,
                    base_global_logit,
                    base_local_score,
                    img_size,
                )
                adaptive_global_logit = base_global_logit
                adaptive_local_score = reaxis_outputs["conditioned_local_map"]
            else:
                if legacy_needs_base:
                    base_global_logit, base_local_score = textual_learner.compute_global_local_score(
                        query_feats,
                        query_patch_feats,
                        learned_text_features,
                        **patch_ms_kwargs_for_branch(args, "base"),
                    )
                    base_local_score = maybe_apply_fb_suppression(
                        args,
                        "base",
                        base_local_score,
                        local_vl_score.detach(),
                    )
                updated_text_features, _, _ = anchor_updater(
                    query_feats.detach().float(),
                    learned_text_features,
                )
                adaptive_global_logit, adaptive_local_score = compute_global_local_score_batchwise(
                    query_feats.float(),
                    [patch_feat.float() for patch_feat in query_patch_feats],
                    updated_text_features,
                    img_size,
                    **patch_ms_kwargs_for_branch(args, "conditioned"),
                )
                adaptive_local_score = maybe_apply_fb_suppression(
                    args,
                    "conditioned",
                    adaptive_local_score,
                    local_vl_score.detach(),
                )

        # ====================== pq_adapter ======================
        if args.pq_learner and k_shots > 0 and not adapter_only_stage:

            global_pq_logit, local_pq_map_list, align_score_list = pq_learner(query_feats, query_patch_feats, prompt_feats, prompt_patch_feats)

            local_pq_map_list = [x[:, 1].unsqueeze(1) for x in local_pq_map_list]
            local_pq_map = torch.concat(local_pq_map_list, dim=1).mean(dim=1).detach()
            align_score = fusion_fun(align_score_list, fusion_type = 'harmonic_mean')[:, 0]

            if isinstance(global_pq_logit, list):
                global_pq_score = [x.softmax(-1).unsqueeze(-1) for x in global_pq_logit]
                global_pq_score = torch.concat(global_pq_score, dim=-1).mean(dim=-1).detach()
                global_pq_score = global_pq_score[:, 1].detach()
            else:
                global_pq_score = global_pq_logit.softmax(-1)
                global_pq_score = global_pq_score[:, 1].detach()

        if stage2_anchor_stage:
            if reaxis_stage_ii:
                branch_pooling_size = (
                    int((query_patch_feats[-1].shape[1] - 1) ** 0.5),
                    int((query_patch_feats[-1].shape[1] - 1) ** 0.5),
                )
                pixel_anomaly_map = reaxis_outputs["pixel_prob_eval"].detach()
                image_anomaly_pred = reaxis_outputs["image_score"].detach()
                fixed_pixel_map = gaussian_smoothing_2d(torch.sigmoid(reaxis_outputs["fixed_local_margin"]), args.sigma).detach()
                adaptive_pixel_map = gaussian_smoothing_2d(torch.sigmoid(reaxis_outputs["conditioned_local_margin"]), args.sigma).detach()
                fused_pixel_map = pixel_anomaly_map
                fixed_image_score = torch.sigmoid(reaxis_outputs["fixed_global_margin"]).detach()
                adaptive_image_score = torch.sigmoid(reaxis_outputs["base_global_margin"]).detach()
                fused_image_score = image_anomaly_pred
                fixed_pixel_max, _ = torch.max(fixed_pixel_map.view(current_batchsize, -1), dim=1)
                adaptive_pixel_max, _ = torch.max(adaptive_pixel_map.view(current_batchsize, -1), dim=1)
                fused_pixel_max, _ = torch.max(pixel_anomaly_map.view(current_batchsize, -1), dim=1)
                fixed_pixel_mean = fixed_pixel_map.view(current_batchsize, -1).mean(dim=1)
                adaptive_pixel_mean = adaptive_pixel_map.view(current_batchsize, -1).mean(dim=1)
                fused_pixel_mean = pixel_anomaly_map.view(current_batchsize, -1).mean(dim=1)
                if args.enable_branch_ablation_metrics:
                    fixed_score, fixed_map = branch_image_score(
                        reaxis_modules,
                        reaxis_outputs["e_fixed_global"],
                        reaxis_outputs["e_fixed_local"],
                        args.sigma,
                        pooling_size=branch_pooling_size,
                    )
                    base_score, base_map = branch_image_score(
                        reaxis_modules,
                        reaxis_outputs["e_adapt_global"],
                        reaxis_outputs["e_base_local"],
                        args.sigma,
                        pooling_size=branch_pooling_size,
                    )
                    conditioned_score, conditioned_map = branch_image_score(
                        reaxis_modules,
                        reaxis_outputs["e_adapt_global"],
                        reaxis_outputs["e_conditioned_local"],
                        args.sigma,
                        pooling_size=branch_pooling_size,
                    )
                    _, fixed_base_global, fixed_base_local, _ = reaxis_modules.fusion(
                        reaxis_outputs["e_fixed_global"],
                        reaxis_outputs["e_fixed_local"],
                        reaxis_outputs["e_adapt_global"],
                        reaxis_outputs["e_base_local"],
                        reaxis_outputs["e_base_local"],
                    )
                    fixed_base_score, fixed_base_map = branch_image_score(
                        reaxis_modules,
                        fixed_base_global,
                        fixed_base_local,
                        args.sigma,
                        pooling_size=branch_pooling_size,
                    )
                    base_conditioned_score, base_conditioned_map = branch_image_score(
                        reaxis_modules,
                        reaxis_outputs["e_adapt_global"],
                        reaxis_outputs["adaptive_local_margin"],
                        args.sigma,
                        pooling_size=branch_pooling_size,
                    )
                    legacy_global_logit, legacy_local_score = legacy_linear_fusion(
                        global_vl_logit,
                        local_vl_score,
                        reaxis_outputs["conditioned_global_logit"],
                        reaxis_outputs["conditioned_local_map"],
                        args.fixed_adaptive_fuse_weight,
                    )
                    legacy_pixel = gaussian_smoothing_2d(legacy_local_score[:, 1].detach(), args.sigma)
                    legacy_global_score = legacy_global_logit.softmax(-1)[:, 1].detach()
                    legacy_map_max = legacy_pixel.reshape(current_batchsize, -1).max(dim=1).values
                    legacy_image = 0.5 * (legacy_global_score + legacy_map_max)
                    branch_batch_outputs = {
                        "fixed_only": (fixed_score.detach(), fixed_map.detach()),
                        "base_only": (base_score.detach(), base_map.detach()),
                        "conditioned_local_only": (conditioned_score.detach(), conditioned_map.detach()),
                        "fixed_plus_base": (fixed_base_score.detach(), fixed_base_map.detach()),
                        "base_plus_conditioned": (base_conditioned_score.detach(), base_conditioned_map.detach()),
                        "full_STARCLIP": (image_anomaly_pred.detach(), pixel_anomaly_map.detach()),
                        "legacy_linear_fusion": (legacy_image.detach(), legacy_pixel.detach()),
                    }
                    for branch_name, (branch_image, branch_mask) in branch_batch_outputs.items():
                        if cpu_eva:
                            branch_eval_predictions[branch_name]["pr_masks"].append(branch_mask.cpu())
                            branch_eval_predictions[branch_name]["pr_anomalys"].append(branch_image.cpu())
                        else:
                            branch_eval_predictions[branch_name]["pr_masks"].append(branch_mask)
                            branch_eval_predictions[branch_name]["pr_anomalys"].append(branch_image)
            else:
                # Stage I/legacy 2.2 evaluates fused fixed-anchor VisualAdapter and adaptive textual-anchor branches.
                fuse_weight = args.fixed_adaptive_fuse_weight
                fused_global_logit, fused_local_score = dual_anchor_margin_fusion(
                    global_vl_logit,
                    local_vl_score,
                    adaptive_global_logit,
                    adaptive_local_score,
                    fuse_weight,
                    mode=args.dual_anchor_margin_fusion_mode,
                    base_global_logit=base_global_logit,
                    base_local_score=base_local_score,
                    alpha_global=args.legacy_alpha_global,
                    alpha_local=args.legacy_alpha_local,
                )
                fused_local_score = maybe_apply_fb_suppression(
                    args,
                    "final",
                    fused_local_score,
                    local_vl_score.detach(),
                )
                local_score_for_image = fused_local_score
                if source_memory_bank is not None:
                    memory_probability = source_memory_anomaly_probability(
                        query_patch_feats[-1],
                        source_memory_bank["normal"],
                        source_memory_bank["anomaly"],
                        topk=args.source_memory_topk,
                        temperature=args.source_memory_temperature,
                        chunk_size=args.source_memory_chunk_size,
                        output_size=fused_local_score.shape[-2:],
                    )
                    fused_local_score = apply_source_memory_residual(
                        fused_local_score,
                        memory_probability,
                        weight=args.source_memory_weight,
                    )
                    if args.source_memory_for_image_score:
                        local_score_for_image = fused_local_score

                fixed_pixel_map = local_vl_score[:, 1].detach()
                adaptive_pixel_map = adaptive_local_score[:, 1].detach()
                fused_pixel_map_raw = fused_local_score[:, 1].detach()
                local_image_score_raw = local_score_for_image[:, 1].detach()

                fixed_pixel_map = torch.stack([torch.from_numpy(gaussian_filter(i, sigma = args.sigma)) for i in fixed_pixel_map.cpu()], dim = 0).to(device)
                adaptive_pixel_map = torch.stack([torch.from_numpy(gaussian_filter(i, sigma = args.sigma)) for i in adaptive_pixel_map.cpu()], dim = 0).to(device)
                pixel_anomaly_map = torch.stack([torch.from_numpy(gaussian_filter(i, sigma = args.sigma)) for i in fused_pixel_map_raw.cpu()], dim = 0).to(device)
                local_image_score_smoothed = torch.stack(
                    [torch.from_numpy(gaussian_filter(i, sigma=args.sigma)) for i in local_image_score_raw.cpu()],
                    dim=0,
                ).to(device)

                fixed_image_score = global_vl_logit.softmax(-1)[:, 1].detach()
                adaptive_image_score = adaptive_global_logit.softmax(-1)[:, 1].detach()
                fused_image_score = fused_global_logit.softmax(-1)[:, 1].detach()

                local_image_score = legacy_local_image_score(
                    local_image_score_raw,
                    local_image_score_smoothed,
                    args.legacy_image_score_local_pooling,
                    args.legacy_image_score_topk_ratio,
                )
                legacy_score_fusion = args.fusion_type if args.fusion_type in ("average_mean", "harmonic_mean") else args.legacy_score_fusion_type
                image_anomaly_pred = fusion_fun([fused_image_score, local_image_score], fusion_type = legacy_score_fusion)

                fixed_pixel_max, _ = torch.max(fixed_pixel_map.view(current_batchsize, -1), dim=1)
                adaptive_pixel_max, _ = torch.max(adaptive_pixel_map.view(current_batchsize, -1), dim=1)
                fused_pixel_max, _ = torch.max(pixel_anomaly_map.view(current_batchsize, -1), dim=1)
                fixed_pixel_mean = fixed_pixel_map.view(current_batchsize, -1).mean(dim=1)
                adaptive_pixel_mean = adaptive_pixel_map.view(current_batchsize, -1).mean(dim=1)
                fused_pixel_mean = pixel_anomaly_map.view(current_batchsize, -1).mean(dim=1)
            for batch_idx in range(current_batchsize):
                stage2_branch_rows.append({
                    "sample_id": to_scalar_string(sample_id[batch_idx]),
                    "cls_name": str(cls_name[batch_idx]),
                    "query_path": str(query_path[batch_idx]),
                    "gt_anomaly": int(gt_anomaly[batch_idx].detach().cpu().item()),
                    "fixed_image_score": float(fixed_image_score[batch_idx].detach().cpu().item()),
                    "adaptive_image_score": float(adaptive_image_score[batch_idx].detach().cpu().item()),
                    "fused_image_score": float(fused_image_score[batch_idx].detach().cpu().item()),
                    "fixed_pixel_max": float(fixed_pixel_max[batch_idx].detach().cpu().item()),
                    "adaptive_pixel_max": float(adaptive_pixel_max[batch_idx].detach().cpu().item()),
                    "fused_pixel_max": float(fused_pixel_max[batch_idx].detach().cpu().item()),
                    "fixed_pixel_mean": float(fixed_pixel_mean[batch_idx].detach().cpu().item()),
                    "adaptive_pixel_mean": float(adaptive_pixel_mean[batch_idx].detach().cpu().item()),
                    "fused_pixel_mean": float(fused_pixel_mean[batch_idx].detach().cpu().item()),
                })

        elif stage1_fixed_anchor_visual:
            # Stage 1 evaluates only the fixed-anchor VisualAdapter branch.
            pixel_anomaly_map = local_vl_map
            pixel_anomaly_map = torch.stack([torch.from_numpy(gaussian_filter(i, sigma = args.sigma)) for i in pixel_anomaly_map.cpu()], dim = 0)
            pixel_anomaly_map = pixel_anomaly_map.to(device)

            anomaly_map_max, _ = torch.max(pixel_anomaly_map.view(current_batchsize, -1), dim=1)
            image_anomaly_pred = fusion_fun([global_vl_score, anomaly_map_max], fusion_type = args.fusion_type)

        elif k_shots > 0:
            # get pixel level prediction
            pixel_anomaly_map = fusion_fun([local_vl_map, local_tl_map, local_pq_map], fusion_type = args.fusion_type)
            pixel_anomaly_map = fusion_fun([pixel_anomaly_map, align_score], fusion_type = 'harmonic_mean')
            pixel_anomaly_map = torch.stack([torch.from_numpy(gaussian_filter(i, sigma = args.sigma)) for i in pixel_anomaly_map.cpu()], dim = 0)
            pixel_anomaly_map = pixel_anomaly_map.to(device)

            # get image level prediction
            anomaly_map_max, _ = torch.max(pixel_anomaly_map.view(current_batchsize, -1), dim=1)
            image_anomaly_pred = fusion_fun([global_vl_score, global_tl_score, global_pq_score], fusion_type = args.fusion_type)
            image_anomaly_pred = fusion_fun([image_anomaly_pred, anomaly_map_max], fusion_type = "harmonic_mean")

        else:
            # get pixel level prediction
            pixel_anomaly_map = fusion_fun([local_vl_map, local_tl_map], fusion_type = args.fusion_type)

            pixel_anomaly_map = torch.stack([torch.from_numpy(gaussian_filter(i, sigma = args.sigma)) for i in pixel_anomaly_map.cpu()], dim = 0)
            pixel_anomaly_map = pixel_anomaly_map.to(device)

            # get image level prediction
            anomaly_map_max, _ = torch.max(pixel_anomaly_map.view(current_batchsize, -1), dim=1)
            image_anomaly_pred = fusion_fun([global_vl_score, global_tl_score, anomaly_map_max], fusion_type = args.fusion_type)


        if dataset_name in ['Real-IAD-Variety', 'RealIAD', 'bmad-medical']:
            resize_mask = 256
            if resize_mask is not None:
                pixel_anomaly_map = F.interpolate(pixel_anomaly_map[:, None], size=(resize_mask, resize_mask), mode='bilinear', align_corners=False)
                pixel_anomaly_map = pixel_anomaly_map[:, 0]
                if reaxis_stage_ii and args.enable_branch_ablation_metrics:
                    for branch_name in branch_eval_predictions:
                        branch_map = branch_eval_predictions[branch_name]["pr_masks"][-1].to(device)
                        branch_map = F.interpolate(
                            branch_map[:, None],
                            size=(resize_mask, resize_mask),
                            mode='bilinear',
                            align_corners=False,
                        )[:, 0]
                        branch_eval_predictions[branch_name]["pr_masks"][-1] = branch_map.cpu() if cpu_eva else branch_map
                gt_mask = F.interpolate(gt_mask[:, None], size=(resize_mask, resize_mask), mode='nearest')
                gt_mask = gt_mask.bool().int()

        if not torch.isfinite(pixel_anomaly_map).all() or not torch.isfinite(image_anomaly_pred).all():
            raise ValueError("Model predictions contain NaN or infinity; evaluation cannot produce valid metrics")

        if args.save_visuals:
            selected_indices = []
            max_per_class = args.visual_max_samples_per_class
            for batch_idx, batch_cls_name in enumerate(cls_name):
                if max_per_class <= 0 or visual_counts[batch_cls_name] < max_per_class:
                    selected_indices.append(batch_idx)
                    visual_counts[batch_cls_name] += 1
            if selected_indices:
                selected_paths = [query_path[i] for i in selected_indices]
                selected_cls_names = [cls_name[i] for i in selected_indices]
                selected_images = items["img"][selected_indices].detach().cpu()
                selected_masks = items["img_mask"][selected_indices].detach().cpu()
                selected_maps = pixel_anomaly_map[selected_indices].detach().cpu().numpy()
                visualizer(
                    selected_paths,
                    selected_images,
                    selected_maps,
                    (img_size, img_size),
                    args.visual_save_path or save_path,
                    selected_cls_names,
                    selected_masks,
                )
                visual_saved_total += len(selected_indices)

        sample_ids.append(np.array(sample_id))
        cls_names.append(np.array(cls_name))
        query_paths.append(np.array(query_path))
        if cpu_eva:
            gt_masks.append(gt_mask.detach().int().cpu())
            pr_masks.append(pixel_anomaly_map.detach().cpu())

            gt_anomalys.append(gt_anomaly.detach().int().cpu())
            pr_anomalys.append(image_anomaly_pred.detach().cpu())
        else:
            gt_masks.append(gt_mask.detach().int())
            pr_masks.append(pixel_anomaly_map.detach())


            gt_anomalys.append(gt_anomaly.detach().int())
            pr_anomalys.append(image_anomaly_pred.detach())

    # ====================== Evaluation ======================
    results_eval = dict(sample_ids=sample_ids, gt_masks=gt_masks, pr_masks=pr_masks, cls_names=cls_names, gt_anomalys=gt_anomalys, pr_anomalys=pr_anomalys, query_paths=query_paths)
    results_eval = {k: np.concatenate(v, axis=0) if k in ['cls_names', 'query_paths', 'sample_ids']  else torch.cat(v, dim=0) for k, v in results_eval.items()}


    # save results
    msg = {}
    for idx, cls_name in enumerate(tqdm(obj_list)):
        metric_results = evaluator.run(results_eval, cls_name, logger)
        msg['Name'] = msg.get('Name', [])
        msg['Name'].append(cls_name)
        avg_act = True if len(obj_list) > 1 and idx == len(obj_list) - 1 else False
        msg['Name'].append('Avg') if avg_act else None

        for metric in eval_metrics:
            metric_result = metric_results[metric] * 100

            msg[metric] = msg.get(metric, [])
            msg[metric].append(metric_result)

            if avg_act:
                metric_result_avg = sum(msg[metric]) / len(msg[metric])
                msg[metric].append(metric_result_avg)

    tab = tabulate(msg, headers='keys', tablefmt="pipe", floatfmt='.1f', numalign="center", stralign="center", )
    logger.info('\n' + tab)
    os.makedirs(save_path, exist_ok=True)
    metric_rows = []
    for row_idx, name in enumerate(msg.get("Name", [])):
        row = {"Name": name}
        for metric in eval_metrics:
            row[metric] = float(msg[metric][row_idx])
        metric_rows.append(row)
    metrics_json = {
        "metrics_percent": metric_rows,
        "avg_percent": next((row for row in metric_rows if row["Name"] == "Avg"), metric_rows[-1] if metric_rows else {}),
        "eval_metrics": eval_metrics,
        "checkpoint_path": args.checkpoint_path,
        "reaxis_preset": args.reaxis_preset if stage2_visual_anchor_update else None,
        "starclip_preset": args.starclip_preset if stage2_visual_anchor_update else None,
        "dual_anchor_margin_fusion_mode": args.dual_anchor_margin_fusion_mode if stage2_visual_anchor_update else None,
        "legacy_alpha_global": args.legacy_alpha_global,
        "legacy_alpha_local": args.legacy_alpha_local,
        "legacy_image_score_local_pooling": args.legacy_image_score_local_pooling,
        "legacy_image_score_topk_ratio": args.legacy_image_score_topk_ratio,
        "use_source_memory_residual": args.use_source_memory_residual,
        "source_memory_dataset": args.source_memory_dataset,
        "source_memory_data_path": args.source_memory_data_path,
        "source_memory_weight": args.source_memory_weight,
        "source_memory_temperature": args.source_memory_temperature,
        "source_memory_topk": args.source_memory_topk,
        "source_memory_max_normal_patches": args.source_memory_max_normal_patches,
        "source_memory_max_anomaly_patches": args.source_memory_max_anomaly_patches,
        "source_memory_for_image_score": args.source_memory_for_image_score,
        "source_validation_split_path": args.source_validation_split_path,
        "source_validation_role": args.source_validation_role,
        "source_split_hash": stable_source_split_hash(args.source_validation_split_path),
    }
    with open(os.path.join(save_path, "metrics.json"), "w") as f:
        import json
        def json_metric_values(value):
            if isinstance(value, dict):
                return {key: json_metric_values(item) for key, item in value.items()}
            if isinstance(value, list):
                return [json_metric_values(item) for item in value]
            if isinstance(value, (float, np.floating)) and not np.isfinite(value):
                return None
            return value
        json.dump(json_metric_values(metrics_json), f, indent=2, allow_nan=False)
        f.write("\n")
    for csv_name in ["metrics.csv", "per_class_metrics.csv"]:
        with open(os.path.join(save_path, csv_name), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["Name"] + list(eval_metrics))
            writer.writeheader()
            writer.writerows(metric_rows)

    if reaxis_stage_ii and args.enable_branch_ablation_metrics and branch_eval_predictions:
        for branch_name, predictions in branch_eval_predictions.items():
            branch_eval = dict(results_eval)
            branch_eval["pr_masks"] = torch.cat(predictions["pr_masks"], dim=0)
            branch_eval["pr_anomalys"] = torch.cat(predictions["pr_anomalys"], dim=0)
            branch_msg = {}
            for idx, cls_name in enumerate(tqdm(obj_list)):
                metric_results = evaluator.run(branch_eval, cls_name, logger)
                branch_msg['Name'] = branch_msg.get('Name', [])
                branch_msg['Name'].append(cls_name)
                avg_act = True if len(obj_list) > 1 and idx == len(obj_list) - 1 else False
                branch_msg['Name'].append('Avg') if avg_act else None
                for metric in eval_metrics:
                    metric_result = metric_results[metric] * 100
                    branch_msg[metric] = branch_msg.get(metric, [])
                    branch_msg[metric].append(metric_result)
                    if avg_act:
                        branch_msg[metric].append(sum(branch_msg[metric]) / len(branch_msg[metric]))
            branch_tab = tabulate(branch_msg, headers='keys', tablefmt="pipe", floatfmt='.1f', numalign="center", stralign="center")
            logger.info('\nBranch ablation: {}\n{}'.format(branch_name, branch_tab))

    if stage2_anchor_stage and stage2_branch_rows:
        os.makedirs(save_path, exist_ok=True)
        branch_csv_name = "starclip_branch_predictions.csv" if stage2_visual_anchor_update else "stage2_branch_predictions.csv"
        branch_csv_path = os.path.join(save_path, branch_csv_name)
        with open(branch_csv_path, "w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=list(stage2_branch_rows[0].keys()))
            writer.writeheader()
            writer.writerows(stage2_branch_rows)
        logger.info("saved branch predictions to {}".format(branch_csv_path))

    if args.save_visuals:
        logger.info("saved {} visualization images to {}".format(visual_saved_total, args.visual_save_path or save_path))




# Compatibility exports for existing preflight tools and scripts.
load_starclip_modules_from_checkpoint = load_reaxis_modules_from_checkpoint
starclip_stage_ii_forward = reaxis_stage_ii_forward

if __name__ == '__main__':
    parser = argparse.ArgumentParser("ReAxis", add_help=True)
    # paths
    parser.add_argument("--test_data_path", type=str, default="./data/mvtec", help="path to test dataset")
    parser.add_argument("--save_path", type=str, default='./results/', help='path to save results')
    parser.add_argument("--pretrained_model", type=str, default='ViT-L/14@336px', help="pre-trained model name")
    parser.add_argument("--checkpoint_path", type=str, required=True, help='path to a trained checkpoint file')
    # model
    parser.add_argument("--dataset", type=str, default='mvtec')
    parser.add_argument("--source_validation_split_path", type=str, default=None,
                        help="optional source-validation split file for legal checkpoint selection")
    parser.add_argument("--source_validation_role", type=str, default=None, choices=[None, "train", "val"],
                        help="split role to evaluate when source_validation_split_path is set")
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24], help="features used")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--image_size", type=int, default=518, help="image size")
    parser.add_argument("--n_ctx", type=int, default=12, help="zero shot")
    parser.add_argument("--seed", type=int, default=10, help="random seed")
    parser.add_argument("--sigma", type=int, default=4, help="zero shot")
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
                        help="unused at test time; kept for train/test CLI compatibility")
    parser.add_argument("--stage2_checkpoint_path", type=str, default=None,
                        help="unused at test time; kept for train/test CLI compatibility")
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
    parser.add_argument("--legacy_score_fusion_type", type=str, default="average_mean",
                        choices=["average_mean", "harmonic_mean"],
                        help="legacy image-score fusion used when fusion_type is ReAxis-specific")
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
    parser.add_argument("--legacy_image_score_local_pooling", type=str, default="legacy_smoothed_max",
                        choices=[
                            "legacy_smoothed_max",
                            "raw_max",
                            "raw_topk_mean",
                            "smoothed_max",
                            "smoothed_topk_mean",
                        ],
                        help="local map pooling used for legacy image-level score")
    parser.add_argument("--legacy_image_score_topk_ratio", type=float, default=0.01,
                        help="top-k ratio used by legacy raw/smoothed topk image-score pooling")
    parser.add_argument("--use_source_memory_residual", type=str2bool, nargs="?", const=True, default=False,
                        help="enable source-only CLIP patch memory residual for Stage II local maps")
    parser.add_argument("--source_memory_data_path", type=str, default=None,
                        help="source dataset root used to build labeled source patch memory")
    parser.add_argument("--source_memory_dataset", type=str, default=None,
                        help="source dataset name used to build labeled source patch memory")
    parser.add_argument("--source_memory_batch_size", type=int, default=8,
                        help="batch size for source patch memory construction")
    parser.add_argument("--source_memory_max_normal_patches", type=int, default=4096,
                        help="maximum normal source patches kept in memory")
    parser.add_argument("--source_memory_max_anomaly_patches", type=int, default=4096,
                        help="maximum anomaly source patches kept in memory")
    parser.add_argument("--source_memory_weight", type=float, default=0.1,
                        help="blend weight for source-memory local probability residual")
    parser.add_argument("--source_memory_temperature", type=float, default=10.0,
                        help="temperature applied to abnormal-vs-normal source memory evidence")
    parser.add_argument("--source_memory_topk", type=int, default=5,
                        help="top-k source memory neighbors averaged for normal/anomaly evidence")
    parser.add_argument("--source_memory_chunk_size", type=int, default=4096,
                        help="target patch chunk size for source memory similarity computation")
    parser.add_argument("--source_memory_for_image_score", type=str2bool, nargs="?", const=True, default=True,
                        help="when false, source-memory residual affects pixel map only and image local pooling uses the pre-memory local map")
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
                        help="unused at test time; kept for train/test CLI compatibility")
    parser.add_argument("--scr_loss_weight", type=float, default=0.0,
                        help="unused at test time; kept for train/test CLI compatibility")
    parser.add_argument("--scr_loss_source", type=str, default="adaptive",
                        choices=["adaptive", "final", "both", "base"],
                        help="unused at test time; kept for train/test CLI compatibility")
    parser.add_argument("--scr_margin", type=float, default=0.5,
                        help="unused at test time; kept for train/test CLI compatibility")
    parser.add_argument("--anchor_reg_weight", type=float, default=0.01,
                        help="unused at test time; kept for train/test CLI compatibility")
    parser.add_argument("--freeze_visual_adapter_in_stage2", type=str2bool, nargs="?", const=True, default=True,
                        help="unused at test time; kept for train/test CLI compatibility")
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
    parser.add_argument("--condition_residual_cap", type=float, default=4.0,
                        help="conditioned-local bounded residual cap in evidence space")
    parser.add_argument("--global_residual_cap", type=float, default=4.0,
                        help="global bounded residual cap in evidence space")
    parser.add_argument("--local_residual_cap", type=float, default=4.0,
                        help="local bounded residual cap in evidence space")
    parser.add_argument("--condition_gate_init", type=float, default=0.15,
                        help="initial conditioned-local residual gate")
    parser.add_argument("--adapt_global_gate_init", type=float, default=0.4,
                        help="initial Stage II global residual gate")
    parser.add_argument("--adapt_local_gate_init", type=float, default=0.4,
                        help="initial Stage II local residual gate")
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
    parser.add_argument("--delta_reg_weight", type=float, default=0.001,
                        help="unused at test time; kept for train/test CLI compatibility")
    parser.add_argument("--anchor_update_cons_weight", type=float, default=0.05,
                        help="unused at test time; kept for train/test CLI compatibility")
    parser.add_argument("--freeze_textual_adapter_in_stage_ii", "--freeze_textual_adapter_in_stage22",
                        dest="freeze_textual_adapter_in_stage_ii", type=str2bool, nargs="?", const=True, default=True,
                        help="unused at test time; kept for train/test CLI compatibility")
    parser.add_argument("--freeze_visual_adapter_in_stage_ii", "--freeze_visual_adapter_in_stage22",
                        dest="freeze_visual_adapter_in_stage_ii", type=str2bool, nargs="?", const=True, default=True,
                        help="unused at test time; kept for train/test CLI compatibility")
    parser.add_argument("--use_visual_lsar", type=str2bool, nargs="?", const=True, default=False,
                        help="enable LSAR-lite visual residual adapters inside VisualAdapter")
    parser.add_argument("--visual_lsar_bottleneck_ratio", type=int, default=4,
                        help="bottleneck ratio for LSAR-lite visual residual adapters")
    parser.add_argument("--visual_lsar_scale", type=float, default=1.0,
                        help="residual scale for LSAR-lite visual residual adapters")
    parser.add_argument("--train_visual_lsar_when_frozen", type=str2bool, nargs="?", const=True, default=True,
                        help="unused at test time; kept for train/test CLI compatibility")
    parser.add_argument("--visual_learner", action="store_true", help="Enable visual adapter")
    parser.add_argument("--textual_learner", action="store_true", help="Enable textual adapter")
    parser.add_argument("--pq_learner", action="store_true", help="Enable prompt-query adapter")
    parser.add_argument("--eval_metrics", type=str, nargs="+", default=['I-AUROC', 'I-AP', 'P-AUROC', 'P-AUPRO'], help='evaluation metrics')
    parser.add_argument("--fusion_type", type=str, default="average_mean", help='fusion type')
    parser.add_argument("--vl_reduction", type=int, default=4, help="the reduction number of visual learner")
    parser.add_argument("--pq_mid_dim", type=int, default=128, help="the number of the first hidden layer in pqadapter")
    parser.add_argument("--pq_context", action="store_true", help="Enable context feature")
    parser.add_argument("--class_name", type=str, help="class name for a special dataset, for example, bottle in MVTec")
    parser.add_argument("--cpu_eva", type=str2bool, nargs="?", const=True, default=False,
                        help="store collected predictions on CPU to reduce GPU memory; metrics always run on CPU")
    parser.add_argument("--aupro_num_thresholds", type=int, default=None,
                        help="maximum distinct score thresholds for P-AUPRO (default: 200); sampled by score rank")
    parser.add_argument("--enable_branch_ablation_metrics", type=str2bool, nargs="?", const=True, default=False,
                        help="evaluate fixed/base/conditioned/fusion branch ablations for ReAxis")
    parser.add_argument("--save_visuals", type=str2bool, nargs="?", const=True, default=False,
                        help="save AdaptCLIP-style heatmap overlays with GT contours")
    parser.add_argument("--visual_save_path", type=str, default=None,
                        help="optional output root for visualization images; defaults to save_path")
    parser.add_argument("--visual_max_samples_per_class", type=int, default=0,
                        help="maximum visualization images per class; 0 saves all")
    args = parser.parse_args()
    print(args)
    setup_seed(args.seed)
    test(args)
