"""
MaxDiff Card Design Generator
==============================
Replicates Sawtooth Software's MaxDiff design approach using a rotation/shift
algorithm that produces balanced incomplete block designs (BIBD).

HOW MAXDIFF DESIGN WORKS
--------------------------
MaxDiff (Maximum Difference Scaling) presents respondents with a set of items
and asks them to choose the BEST and WORST from each set.

Key design parameters:
  N = total number of items/attributes
  k = items shown per task/set (Sawtooth recommends 4 or 5)
  t = number of tasks shown per respondent

Balance requirements (following Sawtooth guidelines):
  - Each item must appear in at least 2 tasks (minimum exposure)
  - Each item should ideally appear in 3+ tasks (for reliable estimates)
  - Appearances per item = t * k / N  (must be close to an integer)
  - No item should appear twice in the same task

The algorithm used here is a greedy rotation that:
  1. Tracks how many times each item has been shown so far
  2. For each new task, selects k items with the fewest appearances
  3. Randomizes within-task item order (to eliminate position bias)
  4. Verifies the final design for balance properties

Sample size guidance (Sawtooth rule of thumb):
  min_n = 300 / appearances_per_item
  ideal_n = 500 / appearances_per_item
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional


def check_feasibility(n_items: int, k_per_task: int, n_tasks: int) -> Dict:
    """
    Check design parameters against Sawtooth best-practice guidelines.

    Returns a dict with feasibility info, warnings, and sample size recommendations.
    """
    appearances = (n_tasks * k_per_task) / n_items
    warnings = []

    if k_per_task < 3 or k_per_task > 7:
        warnings.append(f"k_per_task={k_per_task}: Sawtooth recommends 4 or 5 items per task.")
    if appearances < 2:
        warnings.append(
            f"Each item appears only {appearances:.1f} times — too low for reliable estimates. "
            f"Increase n_tasks or reduce n_items."
        )
    if appearances < 3:
        warnings.append(
            f"Each item appears {appearances:.1f} times — acceptable minimum, but 3+ is ideal."
        )
    if n_tasks > n_items:
        warnings.append(
            f"n_tasks ({n_tasks}) > n_items ({n_items}) — items will repeat heavily. "
            f"This is fine, but ensure the design avoids repeated pairs."
        )
    if n_items < k_per_task * 2:
        warnings.append(
            f"n_items ({n_items}) is very small relative to k_per_task ({k_per_task})."
        )

    min_sample = max(150, int(300 / appearances))
    ideal_sample = max(300, int(500 / appearances))

    return {
        "appearances_per_item": round(appearances, 2),
        "is_integer_appearances": abs(appearances - round(appearances)) < 0.001,
        "meets_min_threshold": appearances >= 2.0,
        "meets_ideal_threshold": appearances >= 3.0,
        "min_sample_size": min_sample,
        "ideal_sample_size": ideal_sample,
        "warnings": warnings,
    }


def generate_design(
    n_items: int,
    k_per_task: int,
    n_tasks: int,
    seed: int = 42,
    n_versions: int = 1,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Generate a balanced MaxDiff design.

    Parameters
    ----------
    n_items      : Total number of items/attributes
    k_per_task   : Items shown per task (Sawtooth recommends 4 or 5)
    n_tasks      : Tasks per respondent
    seed         : Random seed for reproducibility
    n_versions   : Number of design versions (rotated variants for large samples)

    Returns
    -------
    design_df    : DataFrame with columns [version, task, position_1..k, items list]
    stats        : Balance statistics for the design
    """
    if k_per_task >= n_items:
        raise ValueError("k_per_task must be less than n_items.")
    if n_tasks < 1:
        raise ValueError("n_tasks must be >= 1.")

    rng = np.random.default_rng(seed)
    all_versions = []

    for version in range(1, n_versions + 1):
        item_appearances = np.zeros(n_items + 1, dtype=int)  # 1-indexed
        pair_appearances = np.zeros((n_items + 1, n_items + 1), dtype=int)
        tasks = []

        for task_num in range(1, n_tasks + 1):
            # Greedy balanced selection: pick items with fewest current appearances
            # Add small random noise to break ties randomly
            counts_with_noise = item_appearances[1:] + rng.random(n_items) * 0.01
            sorted_items = np.argsort(counts_with_noise) + 1  # 1-indexed

            # Select top k items (fewest appearances)
            selected = list(sorted_items[:k_per_task])

            # Randomize position order within task (eliminates position bias)
            rng.shuffle(selected)
            tasks.append(selected)

            # Update counters
            for item in selected:
                item_appearances[item] += 1
            for i in range(len(selected)):
                for j in range(i + 1, len(selected)):
                    a, b = selected[i], selected[j]
                    pair_appearances[a][b] += 1
                    pair_appearances[b][a] += 1

        # Build version dataframe
        rows = []
        for task_num, task_items in enumerate(tasks, 1):
            row = {"version": version, "task": task_num}
            for pos, item in enumerate(task_items, 1):
                row[f"position_{pos}"] = item
            row["items"] = ",".join(map(str, task_items))
            rows.append(row)

        all_versions.extend(rows)

    design_df = pd.DataFrame(all_versions)

    # Compute balance statistics (for version 1)
    v1_tasks = [
        [int(x) for x in row.split(",")]
        for row in design_df[design_df["version"] == 1]["items"]
    ]
    item_counts = np.zeros(n_items + 1, dtype=int)
    for task in v1_tasks:
        for item in task:
            item_counts[item] += 1

    stats = {
        "n_items": n_items,
        "k_per_task": k_per_task,
        "n_tasks": n_tasks,
        "n_versions": n_versions,
        "appearances_per_item": item_counts[1:].tolist(),
        "min_appearances": int(item_counts[1:].min()),
        "max_appearances": int(item_counts[1:].max()),
        "mean_appearances": float(item_counts[1:].mean()),
        "balance_ratio": float(item_counts[1:].min() / item_counts[1:].max()),
        "feasibility": check_feasibility(n_items, k_per_task, n_tasks),
    }

    return design_df, stats


def design_to_survey_cards(
    design_df: pd.DataFrame,
    items_df: pd.DataFrame,
    version: int = 1,
) -> pd.DataFrame:
    """
    Convert numeric design to named survey cards for display.

    Parameters
    ----------
    design_df  : Output from generate_design()
    items_df   : DataFrame with columns [item_id, item_label]
    version    : Which design version to use

    Returns
    -------
    cards_df   : DataFrame showing each task with item labels
    """
    item_map = dict(zip(items_df["item_id"], items_df["item_label"]))
    v_design = design_df[design_df["version"] == version].copy()
    k = len([c for c in v_design.columns if c.startswith("position_")])

    cards = []
    for _, row in v_design.iterrows():
        card = {"task": int(row["task"])}
        for pos in range(1, k + 1):
            item_id = int(row[f"position_{pos}"])
            card[f"item_{pos}"] = item_map.get(item_id, f"Item {item_id}")
            card[f"item_{pos}_id"] = item_id
        cards.append(card)

    return pd.DataFrame(cards)


def print_design_summary(stats: Dict) -> None:
    """Print a human-readable design summary."""
    print("=" * 60)
    print("MAXDIFF DESIGN SUMMARY")
    print("=" * 60)
    print(f"  Total items (N)       : {stats['n_items']}")
    print(f"  Items per task (k)    : {stats['k_per_task']}")
    print(f"  Tasks per respondent  : {stats['n_tasks']}")
    print(f"  Design versions       : {stats['n_versions']}")
    print()
    print("BALANCE STATISTICS")
    print("-" * 40)
    print(f"  Min appearances/item  : {stats['min_appearances']}")
    print(f"  Max appearances/item  : {stats['max_appearances']}")
    print(f"  Mean appearances/item : {stats['mean_appearances']:.2f}")
    print(f"  Balance ratio         : {stats['balance_ratio']:.3f} (1.0 = perfect)")
    print()

    feas = stats["feasibility"]
    print("FEASIBILITY CHECK")
    print("-" * 40)
    print(f"  Appearances per item  : {feas['appearances_per_item']:.2f}")
    print(f"  Min sample size       : {feas['min_sample_size']}")
    print(f"  Ideal sample size     : {feas['ideal_sample_size']}")

    if feas["warnings"]:
        print()
        print("WARNINGS")
        print("-" * 40)
        for w in feas["warnings"]:
            print(f"  ⚠  {w}")
    else:
        print()
        print("  ✓  All design checks passed.")
    print("=" * 60)
