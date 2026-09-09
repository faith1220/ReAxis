"""Source-validation split and selection helpers."""

import json
import hashlib
import math
import os
import random
from datetime import datetime

def canonical_json(data):
    return json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def semantic_split_payload(split):
    keys = [
        "dataset",
        "mode",
        "ratio",
        "seed",
        "train_classes",
        "val_classes",
        "train_img_paths",
        "val_img_paths",
    ]
    return {key: split.get(key) for key in keys if key in split}


COMPOSITE_WEIGHTS = {
    "I-AUROC": 0.30,
    "I-AP": 0.20,
    "P-AUROC": 0.15,
    "P-AUPRO": 0.20,
    "P-AP": 0.15,
}

MVTec_TEXTURE_CLASSES = {"carpet", "grid", "leather", "tile", "wood"}


def _category_type(dataset_name, cls_name):
    if dataset_name == "mvtec":
        return "texture" if cls_name in MVTec_TEXTURE_CLASSES else "object"
    return "unknown"


def _stratified_category_holdout(classes, dataset_name, val_count, rng):
    typed = {
        "texture": [cls_name for cls_name in classes if _category_type(dataset_name, cls_name) == "texture"],
        "object": [cls_name for cls_name in classes if _category_type(dataset_name, cls_name) == "object"],
    }
    if not typed["texture"] or not typed["object"]:
        return None
    if dataset_name == "mvtec" and val_count == 3:
        texture_count = 1
    else:
        texture_count = max(1, int(round(val_count * len(typed["texture"]) / float(len(classes)))))
    texture_count = min(texture_count, len(typed["texture"]), val_count - 1)
    object_count = val_count - texture_count
    if object_count <= 0 or object_count > len(typed["object"]):
        return None
    return sorted(rng.sample(typed["texture"], texture_count) + rng.sample(typed["object"], object_count))


def build_source_validation_split(
    root,
    dataset_name,
    mode="category_holdout",
    ratio=0.2,
    seed=111,
    output_dir=None,
    val_classes=None,
):
    meta_path = os.path.join(root, "meta.json")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError("source meta.json not found: {}".format(meta_path))
    with open(meta_path, "r") as f:
        meta = json.load(f)
    source_items = meta["test"]
    classes = sorted(source_items.keys())
    rng = random.Random(seed)
    output_dir = output_dir or os.path.join(root, "source_validation")
    os.makedirs(output_dir, exist_ok=True)

    split = {
        "dataset": dataset_name,
        "root": os.path.abspath(root),
        "mode_requested": mode,
        "ratio": float(ratio),
        "seed": int(seed),
        "created_at": datetime.utcnow().isoformat() + "Z",
        "source_label_split": "meta.json:test",
    }

    requested_val_classes = []
    if val_classes:
        requested_val_classes = sorted([cls_name.strip() for cls_name in val_classes if cls_name.strip()])
        unknown = sorted(set(requested_val_classes) - set(classes))
        if unknown:
            raise ValueError("validation classes not found in {}: {}".format(dataset_name, unknown))

    if requested_val_classes:
        val_classes = requested_val_classes
        train_classes = [cls_name for cls_name in classes if cls_name not in set(val_classes)]
        split["mode"] = "explicit_category_holdout"
        split["train_classes"] = train_classes
        split["val_classes"] = val_classes
        split["train_img_paths"] = []
        split["val_img_paths"] = []
    elif mode in ["category_holdout", "stratified_category_holdout"] and len(classes) >= 3:
        val_count = max(1, int(round(len(classes) * float(ratio))))
        val_count = min(val_count, len(classes) - 1)
        if mode == "stratified_category_holdout":
            val_classes = _stratified_category_holdout(classes, dataset_name, val_count, rng)
            if val_classes is None:
                val_classes = sorted(rng.sample(classes, val_count))
                split["stratification_fallback"] = "category_holdout"
            else:
                split["category_type"] = {cls_name: _category_type(dataset_name, cls_name) for cls_name in classes}
        else:
            val_classes = sorted(rng.sample(classes, val_count))
        train_classes = [cls_name for cls_name in classes if cls_name not in set(val_classes)]
        split["mode"] = mode
        split["train_classes"] = train_classes
        split["val_classes"] = val_classes
        split["train_img_paths"] = []
        split["val_img_paths"] = []
    else:
        split["mode"] = "image_split"
        train_img_paths = []
        val_img_paths = []
        for cls_name in classes:
            paths = [item["img_path"] for item in source_items[cls_name]]
            rng.shuffle(paths)
            val_count = max(1, int(math.ceil(len(paths) * float(ratio))))
            val_count = min(val_count, max(len(paths) - 1, 1))
            val_set = set(paths[:val_count])
            val_img_paths.extend(sorted(val_set))
            train_img_paths.extend([path for path in paths if path not in val_set])
        split["train_classes"] = classes
        split["val_classes"] = classes
        split["train_img_paths"] = sorted(train_img_paths)
        split["val_img_paths"] = sorted(val_img_paths)

    filename = "{}_source_validation_{}_ratio{}_seed{}.json".format(
        dataset_name,
        split["mode"],
        str(float(ratio)).replace(".", "p"),
        seed,
    )
    path = os.path.join(output_dir, filename)
    if os.path.isfile(path):
        with open(path, "r") as f:
            existing = json.load(f)
        if semantic_split_payload(existing) != semantic_split_payload(split):
            raise ValueError(
                "refusing to overwrite existing source-validation split with different semantics: {}".format(path)
            )
        return path, existing
    split["split_hash"] = sha256_text(canonical_json({k: v for k, v in split.items() if k != "split_hash"}))
    with open(path, "w") as f:
        json.dump(split, f, indent=2)
        f.write("\n")
    return path, split


def validation_composite(metrics):
    score = 0.0
    missing = []
    for metric, weight in COMPOSITE_WEIGHTS.items():
        if metric not in metrics:
            missing.append(metric)
            continue
        value = float(metrics[metric])
        if value < -1e-6 or value > 1.0 + 1e-6:
            raise ValueError("metric {} must be in [0,1] for source composite, got {}".format(metric, value))
        score += weight * min(max(value, 0.0), 1.0)
    if missing:
        raise KeyError("missing metrics for validation composite: {}".format(missing))
    return score


def main():
    import argparse

    parser = argparse.ArgumentParser("Create source-validation split")
    parser.add_argument("--root", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--mode", default="category_holdout", choices=["category_holdout", "stratified_category_holdout", "image_split"])
    parser.add_argument("--ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--val_classes", default=None, help="comma-separated explicit validation classes")
    args = parser.parse_args()
    path, split = build_source_validation_split(
        args.root,
        args.dataset,
        mode=args.mode,
        ratio=args.ratio,
        seed=args.seed,
        output_dir=args.output_dir,
        val_classes=args.val_classes.split(",") if args.val_classes else None,
    )
    print(json.dumps({"path": path, "split": split}, indent=2))


if __name__ == "__main__":
    main()
