"""STAR-CLIP backbone and adapter modules."""

from adaptcliplib.adaptclip import (PQAdapter, STARCLIPBackbone, TextualAdapter,
                                    VisionConditionedAnchorUpdater,
                                    VisualAdapter, VisualResidualAdapter,
                                    aggregate_patch_tokens_multiscale,
                                    compute_global_local_score_batchwise,
                                    fusion_fun)

__all__ = [
    "STARCLIPBackbone",
    "TextualAdapter",
    "VisualAdapter",
    "VisualResidualAdapter",
    "VisionConditionedAnchorUpdater",
    "PQAdapter",
    "aggregate_patch_tokens_multiscale",
    "compute_global_local_score_batchwise",
    "fusion_fun",
]
