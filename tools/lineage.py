"""Checkpoint lineage helpers for source-only HPRF experiments."""

import hashlib
import json
import os
import subprocess
from datetime import datetime


LINEAGE_KEY = "hprf_lineage"


def canonical_json(data):
    return json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path, chunk_size=1024 * 1024):
    if not path or not os.path.isfile(path):
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def stable_source_split_hash(path):
    if not path or not os.path.isfile(path):
        return None
    try:
        payload = read_json(path)
    except Exception:
        return sha256_file(path)
    if isinstance(payload, dict) and payload.get("split_hash"):
        return payload["split_hash"]
    return sha256_file(path)


def git_commit(repo_root=None):
    repo_root = repo_root or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def dataset_classes_from_meta(root):
    meta_path = os.path.join(root, "meta.json")
    if not os.path.isfile(meta_path):
        return []
    meta = read_json(meta_path)
    return sorted(meta.get("test", {}).keys())


def source_split_metadata(args):
    split_path = getattr(args, "source_validation_split_path", None)
    mode = getattr(args, "source_validation_mode", "none")
    ratio = getattr(args, "source_validation_ratio", None)
    seed = getattr(args, "source_validation_seed", None)
    if split_path:
        split_abs = os.path.abspath(split_path)
        split = read_json(split_abs)
        return {
            "source_validation_mode": split.get("mode", mode),
            "source_validation_seed": split.get("seed", seed),
            "source_validation_ratio": split.get("ratio", ratio),
            "source_train_classes": sorted(split.get("train_classes", [])),
            "source_val_classes": sorted(split.get("val_classes", [])),
            "source_split_file": split_abs,
            "source_split_hash": stable_source_split_hash(split_abs),
            "trained_on_full_source": False,
        }
    classes = dataset_classes_from_meta(getattr(args, "train_data_path", ""))
    return {
        "source_validation_mode": mode,
        "source_validation_seed": seed,
        "source_validation_ratio": ratio,
        "source_train_classes": classes,
        "source_val_classes": [],
        "source_split_file": None,
        "source_split_hash": None,
        "trained_on_full_source": mode == "none",
    }


def config_hash(args):
    payload = vars(args).copy()
    # Checkpoint lineage records parent hashes separately from tunable config.
    for key in ["save_path", "stage1_checkpoint_path", "stage2_checkpoint_path", "checkpoint_path"]:
        payload.pop(key, None)
    return sha256_text(canonical_json(payload))


def build_checkpoint_lineage(args, train_stage, parent_checkpoint_path=None):
    split_meta = source_split_metadata(args)
    parent_hash = sha256_file(parent_checkpoint_path) if parent_checkpoint_path else None
    return {
        **split_meta,
        "dataset_name": getattr(args, "dataset", None),
        "train_stage": train_stage,
        "config_hash": config_hash(args),
        "git_commit": git_commit(),
        "created_at": datetime.utcnow().isoformat() + "Z",
        "parent_checkpoint_path": os.path.abspath(parent_checkpoint_path) if parent_checkpoint_path else None,
        "parent_checkpoint_hash": parent_hash,
    }


def lineage_from_checkpoint(checkpoint):
    if not isinstance(checkpoint, dict):
        return None
    return checkpoint.get(LINEAGE_KEY)


def _norm_list(values):
    return sorted(list(values or []))


def validate_parent_lineage(
    args,
    checkpoint,
    checkpoint_path,
    logger=None,
    allow_mismatch=False,
    require_known=True,
):
    """Validate that parent checkpoint was trained on the same source split."""

    lineage = lineage_from_checkpoint(checkpoint)
    if lineage is None:
        message = "checkpoint has no HPRF lineage metadata: {}".format(checkpoint_path)
        if require_known and not allow_mismatch:
            raise ValueError(message)
        if logger is not None:
            logger.warning(message + "; continuing only because mismatch override/legacy mode is enabled")
        return False

    expected = source_split_metadata(args)
    mismatches = []
    if lineage.get("dataset_name") != getattr(args, "dataset", None):
        mismatches.append(("dataset_name", lineage.get("dataset_name"), getattr(args, "dataset", None)))
    for key in ["source_validation_mode", "source_split_hash", "trained_on_full_source"]:
        if lineage.get(key) != expected.get(key):
            mismatches.append((key, lineage.get(key), expected.get(key)))
    for key in ["source_train_classes", "source_val_classes"]:
        if _norm_list(lineage.get(key)) != _norm_list(expected.get(key)):
            mismatches.append((key, _norm_list(lineage.get(key)), _norm_list(expected.get(key))))

    if mismatches:
        message = "checkpoint lineage mismatch for {}: {}".format(checkpoint_path, mismatches)
        if not allow_mismatch:
            raise ValueError(message)
        if logger is not None:
            logger.warning(message + "; continuing because --allow_checkpoint_lineage_mismatch=true")
        return False
    if logger is not None:
        logger.info("checkpoint lineage validated for {}".format(checkpoint_path))
    return True


def validate_child_parent_hash(child_checkpoint, parent_checkpoint_path, logger=None, allow_mismatch=False):
    lineage = lineage_from_checkpoint(child_checkpoint)
    if lineage is None or not parent_checkpoint_path:
        return True
    actual_hash = sha256_file(parent_checkpoint_path)
    recorded_hash = lineage.get("parent_checkpoint_hash")
    if recorded_hash and actual_hash and recorded_hash != actual_hash:
        message = (
            "child checkpoint parent hash mismatch: recorded={}, actual={}, parent={}".format(
                recorded_hash,
                actual_hash,
                parent_checkpoint_path,
            )
        )
        if not allow_mismatch:
            raise ValueError(message)
        if logger is not None:
            logger.warning(message + "; continuing because mismatch override is enabled")
        return False
    return True
