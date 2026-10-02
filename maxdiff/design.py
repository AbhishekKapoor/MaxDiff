"""
MaxDiff Card Design Generator
==============================
Replicates Sawtooth Software's MaxDiff design approach.

ORTHOGONALITY IN MAXDIFF
--------------------------
Two levels of "balance" matter:

1. ORTHOGONAL DESIGN (Balanced Incomplete Block Design — BIBD)
   - Each item appears exactly r times across all tasks
   - Every PAIR of items appears together exactly λ times
   - This is the strongest form of balance
   - Only achievable when the following integer conditions hold:
       r = λ(N−1)/(k−1)  must be an integer
       b = λN(N−1)/(k(k−1)) must be an integer
       b ≥ N  (Fisher's inequality)
   - Sawtooth uses BIBD when possible; otherwise falls back to near-orthogonal

2. NEAR-ORTHOGONAL (Balanced, Non-BIBD) DESIGN
   - Each item appears exactly r times  (item balance ✓)
   - Pairs appear together ⌊λ⌋ or ⌈λ⌉ times  (pair balance ≈)
   - Used when BIBD conditions cannot be met for given (N, k)
   - Correlates > 0.95 with BIBD results in practice

AUTOMATIC PARAMETER SELECTION
-------------------------------
Given N attributes and sample size:
  k  = 5  if N > 10, else 4   (Sawtooth recommendation)
  t  = ⌈3 × N / k⌉            (target ~3 appearances per item)
  min_n = 300 / appearances    (Sawtooth sample size rule)

HOW THE GREEDY ROTATION ALGORITHM WORKS
-----------------------------------------
Step 1 — Initialise: set appearance counter to 0 for every item.
Step 2 — For each task: sort items by current appearance count (ascending),
          break ties randomly to prevent position-order bias.
Step 3 — Select the k items with the fewest appearances.
Step 4 — Randomise within-task item order (eliminates left-right bias in survey).
Step 5 — Update counters; repeat for all t tasks.

When a true BIBD is achievable, the cyclic BIBD construction is used instead,
which guarantees pairwise orthogonality.
"""

import numpy as np
import pandas as pd
from math import gcd
from itertools import combinations
from typing import List, Tuple, Dict, Optional


# ---------------------------------------------------------------------------
# BIBD feasibility check
# ---------------------------------------------------------------------------

def check_bibd_feasibility(n_items: int, k_per_task: int, lambda_val: int = 1) -> Dict:
    """
    Check whether a BIBD exists for given parameters.

    A BIBD(v, k, λ) exists (necessary conditions) when:
      r = λ(v−1)/(k−1)  is a positive integer
      b = λv(v−1)/(k(k−1))  is a positive integer  and  b ≥ v

    These are necessary but not always sufficient — some parameter sets
    satisfy these conditions but no BIBD is known to exist.

    Returns a dict with feasibility info and recommended n_tasks.
    """
    v, k, lam = n_items, k_per_task, lambda_val

    r_num = lam * (v - 1)
    r_den = k - 1
    b_num = lam * v * (v - 1)
    b_den = k * (k - 1)

    r_is_int = r_num % r_den == 0
    b_is_int = b_num % b_den == 0

    r = r_num / r_den
    b = b_num / b_den

    is_feasible = r_is_int and b_is_int and b >= v

    return {
        "lambda": lam,
        "r_appearances": r,       # appearances per item
        "b_tasks": b,             # number of tasks (blocks)
        "r_is_integer": r_is_int,
        "b_is_integer": b_is_int,
        "fishers_inequality": b >= v if (r_is_int and b_is_int) else False,
        "bibd_necessary_conditions_met": is_feasible,
        "n_tasks_needed": int(b) if is_feasible else None,
    }


def find_best_bibd(n_items: int, k_per_task: int, max_tasks: int = 24) -> Optional[Dict]:
    """
    Search over λ = 1, 2, 3, ... to find the smallest BIBD that:
      (a) satisfies necessary integer conditions,
      (b) fits within max_tasks, AND
      (c) has a known cyclic construction we can actually generate.

    Returns the first constructible BIBD params, or None.
    """
    for lam in range(1, 10):
        info = check_bibd_feasibility(n_items, k_per_task, lam)
        if (info["bibd_necessary_conditions_met"]
                and info["n_tasks_needed"] is not None
                and info["n_tasks_needed"] <= max_tasks):
            # Only claim orthogonal if we have the actual construction
            key = (n_items, k_per_task, lam)
            if key in _KNOWN_DIFF_SETS:
                info["found"] = True
                return info
    return None


# ---------------------------------------------------------------------------
# Automatic parameter selection
# ---------------------------------------------------------------------------

def auto_design_params(n_items: int, n_respondents: int) -> Dict:
    """
    Automatically select optimal MaxDiff design parameters.

    Given the number of items and respondents:
      - Chooses k_per_task (4 or 5, following Sawtooth guideline)
      - Determines n_tasks for ~3 appearances per item (ideal) or 2 (minimum)
      - Checks if a BIBD is achievable and uses it if so
      - Reports orthogonality status

    Returns a dict with recommended parameters and design type.
    """
    # Choose k
    k = 5 if n_items > 10 else 4

    # Try BIBD first
    bibd = find_best_bibd(n_items, k, max_tasks=24)

    if bibd and bibd["n_tasks_needed"] is not None:
        n_tasks = bibd["n_tasks_needed"]
        appearances = bibd["r_appearances"]
        design_type = "Orthogonal (BIBD)"
        lambda_val = bibd["lambda"]
        orthogonal = True
    else:
        # Near-orthogonal: target 3 appearances per item
        appearances_target = 3
        n_tasks = int(np.ceil(appearances_target * n_items / k))
        appearances = (n_tasks * k) / n_items
        design_type = "Near-Orthogonal (Balanced)"
        lambda_val = None
        orthogonal = False
        bibd = None

    # Sample size guidance
    min_n = max(150, int(300 / appearances))
    ideal_n = max(300, int(500 / appearances))
    n_ok = n_respondents >= min_n
    n_ideal = n_respondents >= ideal_n

    return {
        "n_items": n_items,
        "k_per_task": k,
        "n_tasks": n_tasks,
        "appearances_per_item": round(appearances, 2),
        "design_type": design_type,
        "is_orthogonal": orthogonal,
        "lambda": lambda_val,
        "bibd_info": bibd,
        "n_respondents": n_respondents,
        "min_n_required": min_n,
        "ideal_n": ideal_n,
        "sample_size_ok": n_ok,
        "sample_size_ideal": n_ideal,
        "warnings": _build_warnings(n_items, k, n_tasks, appearances, n_respondents, min_n, ideal_n),
    }


def _build_warnings(n_items, k, n_tasks, appearances, n_respondents, min_n, ideal_n):
    warnings = []

    # Items per set check (Sawtooth p. 95-96)
    if k > n_items / 2:
        warnings.append(
            f"Items per set ({k}) exceeds half the total items ({n_items}). "
            f"Sawtooth recommends k ≤ {n_items // 2} for this study size."
        )

    # Appearances per item check (Sawtooth Design Guide p. 91-134)
    if appearances < 2:
        warnings.append(
            f"Only {appearances:.1f} appearances/item — below minimum (2.0). "
            f"Increase n_tasks or reduce k for reliable estimates."
        )
    elif appearances < 3:
        warnings.append(
            f"{appearances:.1f} appearances/item — acceptable but 3-5 is ideal "
            f"for HB individual-level scores. For aggregate analysis (Counting/Logit) this is OK."
        )

    # Task fatigue check (Sawtooth Design Guide p. 140-143)
    if n_tasks > 25:
        warnings.append(
            f"{n_tasks} tasks per respondent exceeds Sawtooth recommendation (≤25). "
            f"More tasks risk respondent fatigue and lower data quality."
        )

    # Sample size check
    if n_respondents < min_n:
        warnings.append(
            f"n={n_respondents} is below minimum ({min_n}). "
            f"Estimates will be unreliable. Recommend n≥{ideal_n}."
        )
    elif n_respondents < ideal_n:
        warnings.append(f"n={n_respondents} is below ideal ({ideal_n}) for high-precision estimates.")

    return warnings


# ---------------------------------------------------------------------------
# BIBD generation (cyclic method)
# ---------------------------------------------------------------------------

def _cyclic_bibd(v: int, k: int, diff_sets: List[List[int]]) -> List[List[int]]:
    """
    Generate a cyclic BIBD from a difference set.
    Each starter block is developed mod v to produce v blocks.
    """
    blocks = []
    for starter in diff_sets:
        for shift in range(v):
            block = sorted([(x + shift) % v + 1 for x in starter])
            blocks.append(block)
    # Deduplicate
    seen = set()
    unique = []
    for b in blocks:
        key = tuple(b)
        if key not in seen:
            seen.add(key)
            unique.append(b)
    return unique


# Known difference sets for common MaxDiff parameters (0-indexed, mod v)
# Source: Handbook of Combinatorial Designs
_KNOWN_DIFF_SETS = {
    # (v, k, lambda): list of starter blocks
    (7, 3, 1):  [[0, 1, 3]],
    (7, 4, 2):  [[0, 1, 2, 4]],
    (9, 3, 1):  [[0, 1, 3], [0, 2, 7]],   # resolvable design
    (13, 4, 1): [[0, 1, 3, 9]],
    (13, 3, 1): [[0, 1, 3], [0, 1, 4]],
    (21, 5, 1): [[0, 1, 6, 8, 18]],
}


def _try_cyclic_bibd(n_items: int, k_per_task: int, lambda_val: int) -> Optional[List[List[int]]]:
    """Attempt to generate a BIBD using known difference sets."""
    key = (n_items, k_per_task, lambda_val)
    if key in _KNOWN_DIFF_SETS:
        return _cyclic_bibd(n_items, k_per_task, _KNOWN_DIFF_SETS[key])
    return None


# ---------------------------------------------------------------------------
# Greedy balanced design (near-orthogonal)
# ---------------------------------------------------------------------------

def _greedy_balanced_design(n_items: int, k_per_task: int, n_tasks: int,
                              rng: np.random.Generator) -> List[List[int]]:
    """
    Generate a balanced design using greedy item-appearance minimisation.
    Each task selects the k items with fewest current appearances.
    """
    item_appearances = np.zeros(n_items + 1, dtype=int)
    tasks = []

    for _ in range(n_tasks):
        noise = rng.random(n_items) * 0.01
        counts_with_noise = item_appearances[1:] + noise
        sorted_items = np.argsort(counts_with_noise) + 1  # 1-indexed
        selected = list(sorted_items[:k_per_task])
        rng.shuffle(selected)
        tasks.append(selected)
        for item in selected:
            item_appearances[item] += 1

    return tasks


# ---------------------------------------------------------------------------
# Design statistics
# ---------------------------------------------------------------------------

def _compute_design_stats(tasks: List[List[int]], n_items: int,
                           k_per_task: int, n_tasks: int,
                           n_versions: int, params: Dict) -> Dict:
    """Compute balance and orthogonality statistics for a design."""
    item_counts = np.zeros(n_items + 1, dtype=int)
    pair_counts = np.zeros((n_items + 1, n_items + 1), dtype=int)
    position_counts = np.zeros((n_items + 1, k_per_task + 1), dtype=int)  # position balance

    for task in tasks:
        for pos, item in enumerate(task, 1):
            item_counts[item] += 1
            position_counts[item][pos] += 1
        for a, b in combinations(task, 2):
            pair_counts[a][b] += 1
            pair_counts[b][a] += 1

    # Extract upper-triangle pair counts
    pair_vals = []
    for i in range(1, n_items + 1):
        for j in range(i + 1, n_items + 1):
            pair_vals.append(pair_counts[i][j])

    pair_arr = np.array(pair_vals)
    item_arr = item_counts[1:]

    # Positional balance: CV of position frequencies (Sawtooth p. 213-214)
    position_arr = position_counts[1:, 1:k_per_task+1].flatten()
    position_cv = (position_arr.std() / position_arr.mean()) if position_arr.mean() > 0 else 0

    # Pairwise orthogonality: coefficient of variation of pair counts
    pair_cv = (pair_arr.std() / pair_arr.mean()) if pair_arr.mean() > 0 else 0
    item_cv = (item_arr.std() / item_arr.mean()) if item_arr.mean() > 0 else 0

    # A design is "orthogonal" if pair CV ≈ 0 (all pairs appear equally often)
    is_orthogonal = pair_cv < 0.01

    return {
        "n_items": n_items,
        "k_per_task": k_per_task,
        "n_tasks": n_tasks,
        "n_versions": n_versions,
        "appearances_per_item": item_arr.tolist(),
        "min_appearances": int(item_arr.min()),
        "max_appearances": int(item_arr.max()),
        "mean_appearances": float(item_arr.mean()),
        "item_balance_ratio": float(item_arr.min() / item_arr.max()),
        "pair_counts_min": int(pair_arr.min()),
        "pair_counts_max": int(pair_arr.max()),
        "pair_counts_mean": float(pair_arr.mean()),
        "pair_cv": float(round(pair_cv, 4)),
        "position_cv": float(round(position_cv, 4)),  # Sawtooth positional balance
        "is_orthogonal": is_orthogonal,
        "design_type": params.get("design_type", "Balanced"),
        "feasibility": params,
    }


# ---------------------------------------------------------------------------
# Main public API
# ---------------------------------------------------------------------------

def generate_design_from_attributes(
    attributes: List[str],
    n_respondents: int,
    k_per_task: Optional[int] = None,
    n_tasks: Optional[int] = None,
    seed: int = 42,
    n_versions: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Generate a MaxDiff card design from a list of attribute names and
    the desired sample size. Automatically selects optimal design parameters
    and attempts orthogonal (BIBD) design where possible.

    Parameters
    ----------
    attributes    : List of attribute/item label strings
                    e.g. ['Battery life', 'Camera quality', 'Price / Value']
    n_respondents : Total number of respondents in the survey
    k_per_task    : Items per task (auto-selected if None; Sawtooth recommends 4-5)
    n_tasks       : Tasks per respondent (auto-selected if None)
    seed          : Random seed for reproducibility
    n_versions    : Number of design versions (auto-selected if None per Sawtooth guidance).
                    Sawtooth recommends multiple versions to reduce context bias.
                    Default: min(50, n_respondents) for robust design across respondent pool.

    Returns
    -------
    items_df      : DataFrame [item_id, item_label]
    design_df     : DataFrame [version, task, position_1..k, items]
    stats         : Dict with balance, orthogonality, and feasibility metrics
    """

    # Auto-select n_versions per Sawtooth recommendation (Design Guide p. 125-129)
    if n_versions is None:
        n_versions = min(50, max(6, n_respondents // 50))  # Scale with sample size, cap at 50
    n_items = len(attributes)
    if n_items < 4:
        raise ValueError("MaxDiff requires at least 4 attributes.")

    # Build items DataFrame
    items_df = pd.DataFrame({
        "item_id": list(range(1, n_items + 1)),
        "item_label": attributes,
    })

    # Validate k_per_task if provided (Sawtooth recommendation: k ≤ n_items/2)
    if k_per_task is not None and k_per_task > n_items / 2:
        import warnings as warn_module
        warn_module.warn(
            f"k_per_task ({k_per_task}) exceeds half of n_items ({n_items}). "
            f"Sawtooth recommends k ≤ {n_items // 2}. Proceeding anyway.",
            UserWarning
        )

    # Auto-select parameters
    params = auto_design_params(n_items, n_respondents)
    if k_per_task is not None:
        params["k_per_task"] = k_per_task
    if n_tasks is not None:
        params["n_tasks"] = n_tasks

    # Recompute appearances and warnings after any user overrides
    k = params["k_per_task"]
    t = params["n_tasks"]
    appearances = round(t * k / n_items, 2)
    params["appearances_per_item"] = appearances

    # Regenerate warnings with potentially overridden values
    min_n = max(150, int(300 / appearances))
    ideal_n = max(300, int(500 / appearances))
    params["warnings"] = _build_warnings(n_items, k, t, appearances, n_respondents, min_n, ideal_n)

    if k >= n_items:
        raise ValueError(f"k_per_task ({k}) must be less than n_items ({n_items}).")

    rng = np.random.default_rng(seed)
    all_versions_rows = []
    all_tasks_v1 = None  # for stats

    for version in range(1, n_versions + 1):
        # Attempt BIBD (only for version 1; other versions use rotation)
        tasks = None
        if version == 1 and params.get("is_orthogonal") and params.get("lambda") is not None:
            tasks = _try_cyclic_bibd(n_items, k, params["lambda"])
            if tasks is not None and len(tasks) != t:
                tasks = None  # BIBD size mismatch, fall back

        if tasks is None:
            tasks = _greedy_balanced_design(n_items, k, t, rng)
            if version == 1:
                params["design_type"] = "Near-Orthogonal (Balanced)"
                params["is_orthogonal"] = False

        if version == 1:
            all_tasks_v1 = tasks

        for task_num, task_items in enumerate(tasks, 1):
            row = {"version": version, "task": task_num}
            for pos, item in enumerate(task_items, 1):
                row[f"position_{pos}"] = item
            row["items"] = ",".join(map(str, task_items))
            all_versions_rows.append(row)

    design_df = pd.DataFrame(all_versions_rows)
    stats = _compute_design_stats(all_tasks_v1, n_items, k, t, n_versions, params)

    return items_df, design_df, stats


def generate_design(
    n_items: int,
    k_per_task: int,
    n_tasks: int,
    seed: int = 42,
    n_versions: int = 1,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Legacy API: generate design from numeric parameters.
    Wraps generate_design_from_attributes with placeholder labels.
    """
    attrs = [f"Item {i}" for i in range(1, n_items + 1)]
    _, design_df, stats = generate_design_from_attributes(
        attrs, n_respondents=300,
        k_per_task=k_per_task, n_tasks=n_tasks,
        seed=seed, n_versions=n_versions,
    )
    return design_df, stats


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def design_to_survey_cards(
    design_df: pd.DataFrame,
    items_df: pd.DataFrame,
    version: int = 1,
) -> pd.DataFrame:
    """Convert numeric design to named survey cards for display/export."""
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


def check_feasibility(n_items: int, k_per_task: int, n_tasks: int) -> Dict:
    """Compatibility shim — run feasibility check for given params."""
    appearances = (n_tasks * k_per_task) / n_items
    warnings = []
    if appearances < 2:
        warnings.append(f"Appearances/item={appearances:.2f} < 2 minimum.")
    elif appearances < 3:
        warnings.append(f"Appearances/item={appearances:.2f} < 3 ideal.")
    min_n = max(150, int(300 / appearances))
    return {
        "appearances_per_item": round(appearances, 2),
        "min_sample_size": min_n,
        "ideal_sample_size": max(300, int(500 / appearances)),
        "meets_min_threshold": appearances >= 2.0,
        "meets_ideal_threshold": appearances >= 3.0,
        "warnings": warnings,
    }


def print_design_summary(stats: Dict) -> None:
    """Print a human-readable design summary with orthogonality status."""
    print("=" * 65)
    print("  MAXDIFF CARD DESIGN SUMMARY")
    print("=" * 65)
    feas = stats.get("feasibility", stats)
    print(f"  Total attributes (N)     : {stats['n_items']}")
    print(f"  Attributes per card (k)  : {stats['k_per_task']}")
    print(f"  Cards per respondent (t) : {stats['n_tasks']}")
    print(f"  Design versions          : {stats['n_versions']}")
    print()

    dtype = stats.get("design_type", "?")
    orth = stats.get("is_orthogonal", False)
    orth_sym = "✓ YES" if orth else "✗ NO (near-orthogonal)"
    print(f"  Design type              : {dtype}")
    print(f"  Fully orthogonal (BIBD)  : {orth_sym}")
    if orth:
        lam = feas.get("lambda") if isinstance(feas, dict) else None
        if lam:
            print(f"  λ (pair co-occurrence)   : {lam} (each pair appears exactly {lam} time(s))")
    else:
        mn = stats.get("pair_counts_min", "?")
        mx = stats.get("pair_counts_max", "?")
        cv = stats.get("pair_cv", "?")
        print(f"  Pair co-occurrence range : {mn}–{mx}  (CV={cv:.3f})")

    print()
    print("  ITEM BALANCE")
    print("  " + "-" * 40)
    print(f"  Min appearances/item     : {stats['min_appearances']}")
    print(f"  Max appearances/item     : {stats['max_appearances']}")
    print(f"  Mean appearances/item    : {stats['mean_appearances']:.2f}")
    print(f"  Item balance ratio       : {stats['item_balance_ratio']:.3f}  (1.0 = perfect)")

    print()
    print("  POSITIONAL BALANCE (Sawtooth)")
    print("  " + "-" * 40)
    pos_cv = stats.get('position_cv', 'N/A')
    print(f"  Position coefficient var : {pos_cv:.4f}  (0 = perfect, <0.05 = excellent)")
    if pos_cv != 'N/A' and pos_cv < 0.05:
        print(f"  Status                   : ✓ Excellent (eliminates order bias)")

    # Sample size
    if isinstance(feas, dict):
        min_n = feas.get("min_n_required") or feas.get("min_sample_size")
        ideal_n = feas.get("ideal_n") or feas.get("ideal_sample_size")
        n_resp = feas.get("n_respondents")
        if min_n:
            print()
            print("  SAMPLE SIZE GUIDANCE")
            print("  " + "-" * 40)
            print(f"  Minimum recommended      : n ≥ {min_n}")
            print(f"  Ideal                    : n ≥ {ideal_n}")
            if n_resp:
                ok = "✓" if n_resp >= min_n else "⚠"
                print(f"  Provided                 : n = {n_resp}  {ok}")

        warnings = feas.get("warnings", [])
        if warnings:
            print()
            print("  WARNINGS")
            print("  " + "-" * 40)
            for w in warnings:
                print(f"  ⚠  {w}")
        else:
            print()
            print("  ✓  All design checks passed.")
    print("=" * 65)
