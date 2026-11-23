"""
Parse study_setup.txt or CSV attribute files.
"""

import os
import re
import pandas as pd
from typing import Optional, Dict, Any


_DEFAULTS = {
    "study_name": "MaxDiff Study",
    "n_respondents": 300,
    "k_per_task": None,
    "n_tasks": None,
    "n_versions": 2,
    "method": "both",
    "output": "output/MaxDiff_Results.xlsx",
}


def parse_study_setup(path: str) -> Dict[str, Any]:
    """
    Parse a study_setup.txt file and return a config dict with keys:
        study_name, n_respondents, k_per_task, n_tasks, n_versions,
        method, output, attributes (list of str), items_df (DataFrame).
    """
    cfg = dict(_DEFAULTS)
    attributes = []
    in_attrs = False

    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue

            if line.upper() == "ATTRIBUTES:":
                in_attrs = True
                continue

            if in_attrs:
                attributes.append(line)
                continue

            if "=" in line:
                key, _, val = line.partition("=")
                key = key.strip().upper()
                val = val.strip().strip('"').strip("'")
                _apply_key(cfg, key, val)

    if not attributes:
        raise ValueError(
            f"No attributes found in {path}. "
            "Add an ATTRIBUTES: section with one item per line."
        )

    items_df = pd.DataFrame({
        "item_id": list(range(1, len(attributes) + 1)),
        "item_label": attributes,
    })
    cfg["attributes"] = attributes
    cfg["items_df"] = items_df
    return cfg


def _apply_key(cfg: dict, key: str, val: str) -> None:
    if key == "STUDY_NAME":
        cfg["study_name"] = val
    elif key == "N_RESPONDENTS":
        cfg["n_respondents"] = int(val)
    elif key == "K_PER_TASK":
        cfg["k_per_task"] = int(val)
    elif key == "N_TASKS":
        cfg["n_tasks"] = int(val)
    elif key == "N_VERSIONS":
        cfg["n_versions"] = int(val)
    elif key == "METHOD":
        v = val.lower()
        if v not in ("counts", "logit", "both"):
            raise ValueError(f"METHOD must be counts | logit | both, got: {val}")
        cfg["method"] = v
    elif key == "OUTPUT":
        cfg["output"] = val


def load_items_csv(path: str) -> pd.DataFrame:
    """Load attributes from a CSV with columns item_id, item_label."""
    df = pd.read_csv(path)
    if "item_id" not in df.columns or "item_label" not in df.columns:
        raise ValueError(f"{path}: CSV must have columns item_id, item_label")
    return df[["item_id", "item_label"]].copy()


def find_default_setup(root: str = ".") -> Optional[str]:
    """Return path to input/study_setup.txt if it exists."""
    candidate = os.path.join(root, "input", "study_setup.txt")
    return candidate if os.path.isfile(candidate) else None
