# Alignment with the revised ReAxis manuscript

This document compares the released research implementation with the author-provided revised manuscript, **ReAxis: Normality-Guided Reorientation of the Semantic Decision Axis for Zero-Shot Anomaly Detection**, revision `(3)`. Page numbers refer to that manuscript. The PDF is not included in this source distribution.

This release updates names and documentation while preserving the existing model and training behavior. It does not resolve the algorithmic differences below and does not certify reproduction of the manuscript's tables. Historical checkpoints retain the architecture and training history with which they were produced.

## Terminology and changes in the revised manuscript

The revised manuscript calls Stage II **Normality-Guided Axis Reorientation**. The earlier Stage II heading used **Normality-Guided Axis Calibration**. The revised name is used in the public command alias `stage_ii_normality_guided_axis_reorientation`; older aliases remain accepted.

The revision expands the explanation of the anchor geometry and of source images being treated as queries during Stage II. It retains the two projections, bounded tangent update, symmetric anchor reconstruction, five calibration pairs, hierarchical residual fusion, and training objectives in Eqs. (3)-(11). The disclosed numerical hyperparameters are unchanged. Thus, updating terminology alone does not eliminate the implementation differences identified below.

| Manuscript section | Released symbol or entry |
| --- | --- |
| III-D.1 Query-Derived Normal Context | `build_query_derived_normal_context` |
| III-D.2 Structure-Preserving Axis Reorientation | `StructurePreservingAxisReorientation` |
| III-E.1 Evidence Representation and Calibration | `EvidenceCalibrator` |
| III-E.2 Hierarchical Residual Fusion | `HierarchicalResidualFusion` |
| III-E.3 Global-Local Anomaly Prediction | `GlobalLocalAnomalyPrediction` |
| III-F Stage I: Source-Supervised Base Evidence Learning | `stage_i_source_supervised_base_evidence_learning` |
| III-F Stage II: Normality-Guided Axis Reorientation | `stage_ii_normality_guided_axis_reorientation`, `ReAxisStageIIModules` |

These symbols are available through `reaxislib`. Their implementation is retained under `adaptcliplib`; HPRF/STAR-CLIP names and checkpoint aliases are compatibility interfaces.

## Five implementation differences

### 1. Four independent calibration pairs instead of five

**Manuscript:** Eq. (5), p. 5, and Eq. (11), p. 6, specify independent global calibrators for F and B and local calibrators for F, B, and R. Each pair has its own temperature and shift, with identity initialization. The calibration regularizer sums over `{(F,G), (B,G), (F,L), (B,L), (R,L)}`.

**Current code:** `ReAxisStageIIModules` in `adaptcliplib/reaxis.py` creates `calibrator_fixed_global`, `calibrator_adapt_global`, `calibrator_fixed_local`, and `calibrator_adapt_local`. Its `calibrate` method applies the same `calibrator_adapt_local` to both base-local and conditioned-local evidence. The optimizer groups and identity regularizer likewise cover four independent pairs.

**Consequence:** There are five evidence inputs but four independent calibration parameter pairs. A fifth conditioned-local calibrator would change the architecture and checkpoint state; it has not been silently introduced.

### 2. Last-layer branch scoring despite a four-layer extraction setting

**Manuscript:** The implementation details on p. 6 state that visual layers `{6,12,18,24}` are used for anomaly localization.

**Current code:** The default feature list extracts those layers, but the active branch scorers in `adaptcliplib/adaptclip.py` select the last element:

- `VisualAdapter.forward` uses `patch_features[-1]` for fixed-branch local scoring.
- `TextualAdapter.compute_global_local_score` uses `query_patch_feats[-1]` for base scoring.
- `compute_global_local_score_batchwise` uses `query_patch_feats[-1]` for conditioned scoring.

The normal-context call also uses the last extracted layer. Spatial multiscale aggregation, when enabled, aggregates neighborhoods within that feature tensor; it is not fusion of the four backbone layers.

**Consequence:** Supplying `--features_list 6 12 18 24` does not make the current three branches combine all four layers. The manuscript does not specify an exact cross-layer aggregation rule, so any future correction must document that implementation choice. It also does not explicitly require a separate context estimate from every layer.

### 3. An additional visual-adapter pretraining dependency

**Manuscript:** Fig. 2 and III-F on pp. 4-6 describe two stages. Stage I freezes CLIP and the fixed semantic anchors and optimizes only base prompts. Stage II freezes the fixed/base branches and optimizes the updater, calibrators, and fusion components.

**Current code:** `train.py` requires a `visual_learner` checkpoint before `stage2_learnable_anchor`, the internal name for manuscript Stage I. That checkpoint is produced by `stage1_fixed_anchor_visual`, which trains `VisualAdapter`. The fixed scoring branch then passes global and local CLIP features through those learned visual adapters. Stage I and Stage II freeze them by default, but their earlier training is still part of the model history.

**Consequence:** The current executable workflow is visual pretraining -> Stage I -> Stage II. The README includes all three commands. Removing the dependency or replacing the adapted fixed branch with direct frozen-CLIP scoring would change model behavior, not just terminology.

### 4. Additional loss terms and a different rotation penalty

**Manuscript:** Eqs. (8)-(10) define a joint detection objective using image cross-entropy and pixel focal/Dice losses. Stage I adds the anchor-preservation term. Stage II adds rotation, calibration, and gate regularizers. In particular, p. 6 defines `L_rot = ||d_R - d_B||_2^2`.

**Current code:** The Stage II loss assembly in `train.py` contains separate conditioned-local, adaptive-local, final-local, final-global, and final-image losses. It additionally penalizes the raw updater residual norm. Warm-up uses a subset of these terms. Their nonzero defaults make the objective broader than Eq. (10).

In `StructurePreservingAxisReorientation.forward`, `anchor_rotation_loss` is the squared rotation angle in radians. For unit directions separated by angle `theta`, the manuscript's squared direction difference equals `2 - 2*cos(theta)`, whereas the code uses `theta^2`. They are close for small angles but are not identical. The code also clamps the cosine before `acos` for numerical stability.

**Consequence:** Matching the rotation limit or loss names does not establish mathematical identity of the training objective. The loss weights used by this implementation are not all disclosed in the manuscript. They remain research-code choices rather than newly inferred paper settings.

### 5. Spatial operation order and evaluation smoothing

**Manuscript:** III-E.3, p. 5, describes sigmoid conversion of the fused local evidence at native resolution, strongest-response Top-K pooling on that probability map, and interpolation of the same probability map to the input resolution. The implementation details specify the strongest 1% at the native patch grid.

**Current code:** Branch scoring first produces image-resolution probability maps. Those maps are converted to evidence, calibrated, and fused. `GlobalLocalAnomalyPrediction.forward` downsamples the already fused evidence to the requested native grid before sigmoid and image-level Top-K pooling. It also Gaussian-smooths the full-resolution probability map for pixel evaluation, with the configured sigma. The default image score excludes Gaussian smoothing.

**Consequence:** A native-grid pooling option does not make this sequence identical to fusion on the native grid followed by interpolation. Interpolation, log-odds conversion, nonlinear residual fusion, and sigmoid do not generally commute. Gaussian smoothing is an additional evaluation operation not described in the manuscript's final-map definition.

## Matching components and disclosed settings

The implementation contains the main identifiable ReAxis operations: detached low-abnormality soft weighting, context concatenation `[z_global; z_normal; z_global-z_normal]`, removal of residual components parallel to the base axis and anchor center, bounded tangent reorientation, symmetric anchor reconstruction, conditioned local evidence, bounded hierarchical residual fusion, and a global-local image head.

| Setting | Revised manuscript | Released default or configuration |
| --- | --- | --- |
| Backbone | Frozen CLIP ViT-L/14@336px | Same backbone choice; adapter distinction above |
| Input size | 518 x 518 | 518 |
| Feature layers | 6, 12, 18, 24 | Extracted, but last-layer branch scoring |
| Stage I fixed/base fusion | 0.5 | 0.5 |
| Anchor-preservation weight | 0.01 | 0.01 |
| Normal-context beta | 10 | 10 |
| Maximum rotation | 5 degrees | 5 degrees |
| Conditioned scope | Local only | Local by default |
| Residual bounds R/G/L | 4 / 4 / 4 | 4 / 4 / 4 |
| Initial gates R/G/L/image | 0.15 / 0.4 / 0.4 / 0.5 | Same |
| Calibration temperature range | [0.05, 20] | Same range; sharing distinction above |
| Local image pooling | Strongest 1%, native grid | Same ratio; spatial-order distinction above |
| Optimizer / batch size | Adam / 8 | Adam / 8 |
| Updater learning rate | 1e-3 | 1e-3 in README commands |
| Calibration / fusion learning rates | 1e-4 | 1e-4 in README commands |
| Updater warm-up | First 20% of Stage II | Ratio 0.2; integer epochs rounded up |

The manuscript does not disclose the total epoch count, seed, prompt length, updater hidden size, or every loss weight. The README's 15 epochs and seed 111 are explicit example settings, not attributed paper requirements.

## Data protocol and interpretation of results

The source loader reads `meta['test']` for its supervised image/mask pool even in training mode. This is suitable only when that labeled pool belongs to the auxiliary source benchmark. It must not be confused with permission to train on the target evaluation set. Source and target benchmarks must remain disjoint, and checkpoint or hyperparameter selection must use source data only.

The ReAxis adapter-only inference path bypasses target prompt-memory scoring. Historical `k_shots` arguments and prompt-dataset construction are retained for compatibility. Results should state the source dataset, target dataset, checkpoint, complete configuration, seed, and metric settings.

The public evaluation code was independently rewritten. The documented AUPRO setting uses 200 thresholds, 8-connected ground-truth components, and area up to FPR 0.3 normalized by 0.3. Differences in threshold grids, region connectivity, interpolation, aggregation, or smoothing can change reported values relative to historical implementations. No equivalence to historical logs or the manuscript tables is claimed.

The independent `FocalLoss` rewrite preserves the research code's default binary probability-input and smoothing behavior. It does not remove the objective differences above. A naming change, a compatible checkpoint load, or a software check alone is not evidence of paper-level numerical reproduction.

## Attribution

The release retains GNU GPL version 2 in [LICENSE](../LICENSE), with AdaptCLIP, OpenAI CLIP, and OpenCLIP attribution and applicable notices in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md). The manuscript, weights, datasets, and private experiment logs are not distributed with this source package.
