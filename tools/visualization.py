import os
from pathlib import Path

import cv2
import numpy as np


def normalize_final_map(scoremap, max_value=None, min_value=None):
    scoremap = np.asarray(scoremap, dtype=np.float32)
    lower = float(np.min(scoremap) if min_value is None else min_value)
    upper = float(np.max(scoremap) if max_value is None else max_value)
    if upper - lower < 1e-12:
        return np.zeros_like(scoremap, dtype=np.float32)
    return np.clip((scoremap - lower) / (upper - lower), 0.0, 1.0)


def visualizer(pathes, ori_img, anomaly_map, img_size, save_path, cls_name, img_mask, max=None, min=None):
    del ori_img  # The transformed model tensor is not suitable for paper visualization.

    width, height = int(img_size[0]), int(img_size[1])
    for idx, path in enumerate(pathes):
        defect_type = Path(path).parent.name
        filename = f"{Path(path).stem}.png"
        base_dir = Path(save_path) / "imgs" / cls_name[idx] / defect_type

        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"failed to read visualization source image: {path}")
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)

        final_map = np.asarray(anomaly_map[idx], dtype=np.float32).squeeze()
        if final_map.shape != (height, width):
            final_map = cv2.resize(final_map, (width, height), interpolation=cv2.INTER_LINEAR)
        normalized_map = normalize_final_map(final_map, max_value=max, min_value=min)
        heatmap = cv2.applyColorMap(
            np.round(normalized_map * 255.0).astype(np.uint8),
            cv2.COLORMAP_JET,
        )
        overlay = cv2.addWeighted(image, 0.5, heatmap, 0.5, 0.0)

        gt_mask = np.asarray(img_mask[idx]).squeeze().astype(np.float32)
        if gt_mask.shape != (height, width):
            gt_mask = cv2.resize(gt_mask, (width, height), interpolation=cv2.INTER_NEAREST)
        gt_mask = ((gt_mask > 0.5) * 255).astype(np.uint8)

        outputs = {
            "original": image,
            "gt": gt_mask,
            "heatmap": heatmap,
            "overlay": overlay,
        }
        for output_name, output_image in outputs.items():
            output_dir = base_dir / output_name
            output_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_dir / filename), output_image)

        map_dir = base_dir / "final_map"
        map_dir.mkdir(parents=True, exist_ok=True)
        np.save(str(map_dir / f"{Path(path).stem}.npy"), final_map)


def apply_ad_scoremap(image, scoremap, alpha=0.5):
    normalized_map = normalize_final_map(scoremap)
    heatmap = cv2.applyColorMap(
        np.round(normalized_map * 255.0).astype(np.uint8),
        cv2.COLORMAP_JET,
    )
    return cv2.addWeighted(image, alpha, heatmap, 1.0 - alpha, 0.0)
