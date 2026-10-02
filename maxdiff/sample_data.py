"""
Sample Data Generator for MaxDiff Demonstration
=================================================
Generates realistic synthetic survey responses using a Random Utility Model.

True utilities are predefined for 15 smartphone features, with demographic
variation so the Simulator sheet shows meaningful subgroup differences.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple


# True utilities (population level) — what we expect the model to recover
TRUE_UTILITIES = {
    1:  2.80,  # Price / Value for money
    2:  2.50,  # Battery life
    3:  2.00,  # Camera quality
    4:  1.80,  # Durability / Build quality
    5:  1.60,  # Display quality
    6:  1.50,  # Processing speed
    7:  1.40,  # Water resistance
    8:  1.20,  # 5G / Connectivity
    9:  1.00,  # Screen size
    10: 0.90,  # Biometric security
    11: 0.80,  # Storage capacity
    12: 0.70,  # Operating system ecosystem
    13: 0.60,  # Weight / Portability
    14: 0.50,  # Design / Aesthetics
    15: 0.30,  # Brand reputation
}

# Demographic group offsets on top of the true utilities
# Format: {item_id: delta_utility}
GENDER_DELTAS = {
    "Male": {3: -0.4, 6: 0.3, 8: 0.4, 14: -0.3},       # males care less about camera, more about speed/5G
    "Female": {3: 0.4, 5: 0.3, 14: 0.3, 6: -0.3, 8: -0.4},  # females care more about camera/display
}

AGE_DELTAS = {
    "18-34": {8: 0.5, 6: 0.3, 3: 0.3, 2: -0.3},          # young: tech features
    "35-54": {1: 0.3, 2: 0.3, 7: 0.2, 3: -0.2},           # mid: practical
    "55+":   {2: 0.5, 1: 0.4, 4: 0.3, 8: -0.5, 6: -0.3},  # older: battery/value/durability
}

REGION_DELTAS = {
    "North": {2: 0.2, 4: 0.2, 1: 0.1},
    "South": {3: 0.3, 5: 0.3, 14: 0.2},
    "East":  {6: 0.2, 8: 0.3, 12: 0.2},
    "West":  {1: 0.3, 11: 0.2, 7: 0.2},
}


def _get_respondent_utilities(gender: str, age_group: str, region: str,
                               noise_sd: float = 0.3,
                               rng: np.random.Generator = None) -> dict:
    """Compute respondent-specific utilities (true + group deltas + noise)."""
    if rng is None:
        rng = np.random.default_rng(0)

    utils = {k: v for k, v in TRUE_UTILITIES.items()}

    for item, delta in GENDER_DELTAS.get(gender, {}).items():
        utils[item] = utils.get(item, 0) + delta
    for item, delta in AGE_DELTAS.get(age_group, {}).items():
        utils[item] = utils.get(item, 0) + delta
    for item, delta in REGION_DELTAS.get(region, {}).items():
        utils[item] = utils.get(item, 0) + delta

    # Add respondent-level noise
    for item in utils:
        utils[item] += rng.normal(0, noise_sd)

    return utils


def _choose_best_worst(set_items: List[int], utilities: dict,
                        rng: np.random.Generator,
                        noise_sd: float = 0.5) -> Tuple[int, int]:
    """
    Simulate MaxDiff choice using Random Utility Model.
    Add Gumbel noise to utilities (logit model).
    """
    # Add Gumbel-distributed errors (this IS the logit model)
    scores = {
        item: utilities[item] + rng.gumbel(0, 1) * noise_sd
        for item in set_items
    }
    sorted_items = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    best = sorted_items[0]
    worst = sorted_items[-1]
    return best, worst


def generate_sample_items() -> pd.DataFrame:
    """Return the 15 smartphone features used in this sample study."""
    items = [
        (1,  "Price / Value for money"),
        (2,  "Battery life"),
        (3,  "Camera quality"),
        (4,  "Durability / Build quality"),
        (5,  "Display quality"),
        (6,  "Processing speed"),
        (7,  "Water resistance"),
        (8,  "5G / Connectivity"),
        (9,  "Screen size"),
        (10, "Biometric security"),
        (11, "Storage capacity"),
        (12, "Operating system ecosystem"),
        (13, "Weight / Portability"),
        (14, "Design / Aesthetics"),
        (15, "Brand reputation"),
    ]
    return pd.DataFrame(items, columns=["item_id", "item_label"])


def generate_responses(
    design_df: pd.DataFrame,
    n_respondents: int = 300,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic MaxDiff survey responses.

    Parameters
    ----------
    design_df     : Design DataFrame from design.generate_design()
    n_respondents : Number of simulated respondents
    seed          : Random seed

    Returns
    -------
    responses_df  : Long-format responses with demographics
    """
    rng = np.random.default_rng(seed)

    genders = ["Male", "Female"]
    age_groups = ["18-34", "35-54", "55+"]
    regions = ["North", "South", "East", "West"]

    position_cols = [c for c in design_df.columns if c.startswith("position_")]
    versions = sorted(design_df["version"].unique())
    n_versions = len(versions)

    records = []
    for resp_id in range(1, n_respondents + 1):
        gender = rng.choice(genders, p=[0.48, 0.52])
        age_group = rng.choice(age_groups, p=[0.35, 0.40, 0.25])
        region = rng.choice(regions, p=[0.28, 0.27, 0.23, 0.22])
        version = versions[resp_id % n_versions]

        # Get respondent utilities
        resp_utils = _get_respondent_utilities(gender, age_group, region, rng=rng)

        # Get this respondent's design version
        v_design = design_df[design_df["version"] == version]

        for _, task_row in v_design.iterrows():
            task_num = int(task_row["task"])
            set_items = [int(task_row[pc]) for pc in position_cols]

            best, worst = _choose_best_worst(set_items, resp_utils, rng)

            records.append({
                "respondent_id": resp_id,
                "version": int(version),
                "task": task_num,
                "best_item": best,
                "worst_item": worst,
                "gender": gender,
                "age_group": age_group,
                "region": region,
            })

    return pd.DataFrame(records)
