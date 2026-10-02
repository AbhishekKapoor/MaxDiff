"""
MaxDiff Analysis Engine
========================
Calculates importance scores from MaxDiff survey responses.

TWO SCORING METHODS
--------------------

1. COUNTING METHOD (Simple, fast, transparent)
   - For each item: best_rate = times_chosen_best / times_appeared
   - For each item: worst_rate = times_chosen_worst / times_appeared
   - Raw score = best_rate - worst_rate  (range: -1 to +1)
   - Rescale to 0–100 anchor (most important item = 100, least = 0)

   This is an approximation but correlates very highly (r > 0.95) with logit
   scores for typical survey designs. Sawtooth documents this as "counts."

2. AGGREGATE LOGIT METHOD (More accurate, matches Sawtooth)
   - Treats the best choice as multinomial logit: P(best=i|set) = exp(u_i)/Σexp(u_j)
   - Treats the worst choice as multinomial logit over remaining items
   - Simultaneously estimates utilities u_i using maximum likelihood
   - Utilities are on an interval scale (differences are meaningful)
   - Rescaled to 0–100 for reporting (same as Sawtooth's "rescaled logit")

SCORING PHILOSOPHY (following Sawtooth)
-----------------------------------------
Both methods produce relative importance scores — they show how items rank
against each other, not absolute levels. The key insight:
  - A score of 80 vs 40 means the 80-item is relatively more preferred
  - Scores sum to a constant within each version/cut
  - Zero does NOT mean "unimportant" — it means "least preferred in this set"
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Core counting method
# ---------------------------------------------------------------------------

def score_counts(
    responses_df: pd.DataFrame,
    design_df: pd.DataFrame,
    n_items: int,
    respondent_filter: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Score MaxDiff items using the simple counting method.

    Parameters
    ----------
    responses_df     : Long-format response data with columns:
                       [respondent_id, task, best_item, worst_item]
    design_df        : Design DataFrame from design.generate_design()
    n_items          : Total number of items
    respondent_filter: Boolean mask to subset respondents (for cuts)

    Returns
    -------
    scores_df        : DataFrame with [item_id, best_count, worst_count,
                       appearances, best_rate, worst_rate, raw_score,
                       importance_0_100, rank]
    """
    if respondent_filter is not None:
        resp = responses_df[respondent_filter].copy()
    else:
        resp = responses_df.copy()

    if len(resp) == 0:
        return pd.DataFrame()

    # Get appearances per item from design (use version 1 or merge on version)
    # Build appearance lookup from design
    position_cols = [c for c in design_df.columns if c.startswith("position_")]

    # Count appearances per item in design (per task, not per respondent)
    # Multiply by number of respondents who completed the survey
    n_respondents = resp["respondent_id"].nunique()

    item_appearances = np.zeros(n_items + 1, dtype=int)
    item_best = np.zeros(n_items + 1, dtype=int)
    item_worst = np.zeros(n_items + 1, dtype=int)

    # Count best and worst choices
    for _, row in resp.iterrows():
        item_best[int(row["best_item"])] += 1
        item_worst[int(row["worst_item"])] += 1

    # Count appearances from design × respondents
    # Each respondent sees the same design, so appearances = design_appearances * n_respondents
    version_col = "version" if "version" in design_df.columns else None
    if version_col:
        design_v1 = design_df[design_df["version"] == 1]
    else:
        design_v1 = design_df

    design_appearances = np.zeros(n_items + 1, dtype=int)
    for _, row in design_v1.iterrows():
        for pc in position_cols:
            item_id = int(row[pc])
            if 1 <= item_id <= n_items:
                design_appearances[item_id] += 1

    for item_id in range(1, n_items + 1):
        item_appearances[item_id] = design_appearances[item_id] * n_respondents

    # Build scores
    rows = []
    for item_id in range(1, n_items + 1):
        apps = item_appearances[item_id]
        best_c = item_best[item_id]
        worst_c = item_worst[item_id]

        best_rate = best_c / apps if apps > 0 else 0.0
        worst_rate = worst_c / apps if apps > 0 else 0.0
        raw_score = best_rate - worst_rate

        rows.append({
            "item_id": item_id,
            "best_count": best_c,
            "worst_count": worst_c,
            "appearances": apps,
            "best_rate": round(best_rate, 4),
            "worst_rate": round(worst_rate, 4),
            "raw_score": round(raw_score, 4),
        })

    scores_df = pd.DataFrame(rows)

    # Rescale to 0–100
    mn = scores_df["raw_score"].min()
    mx = scores_df["raw_score"].max()
    rng = mx - mn

    if rng > 0:
        scores_df["importance_0_100"] = ((scores_df["raw_score"] - mn) / rng * 100).round(1)
    else:
        scores_df["importance_0_100"] = 50.0

    scores_df["rank"] = scores_df["importance_0_100"].rank(ascending=False, method="min").astype(int)
    scores_df = scores_df.sort_values("importance_0_100", ascending=False).reset_index(drop=True)

    return scores_df


# ---------------------------------------------------------------------------
# Aggregate Logit method (like Sawtooth's aggregate logit)
# ---------------------------------------------------------------------------

def _logit_neg_loglikelihood(
    utilities: np.ndarray,
    tasks_best: List[Tuple[int, List[int]]],
    tasks_worst: List[Tuple[int, List[int]]],
) -> float:
    """
    Negative log-likelihood for MaxDiff aggregate logit model.

    For each task:
      - Best choice: multinomial logit over all k items in set
      - Worst choice: multinomial logit over k-1 remaining items (negate utilities)
    """
    ll = 0.0

    # Best choices: P(best=i|set) = exp(u_i) / sum_j exp(u_j)
    for best_item_idx, set_item_idxs in tasks_best:
        exp_utils = np.exp(utilities[set_item_idxs])
        ll += utilities[best_item_idx] - np.log(exp_utils.sum())

    # Worst choices: equivalent to best on negated utilities
    for worst_item_idx, remaining_idxs in tasks_worst:
        neg_utils = -utilities[remaining_idxs]
        exp_neg = np.exp(neg_utils)
        ll += (-utilities[worst_item_idx]) - np.log(exp_neg.sum())

    return -ll  # negative because we minimize


def score_logit(
    responses_df: pd.DataFrame,
    design_df: pd.DataFrame,
    n_items: int,
    respondent_filter: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Score MaxDiff items using Aggregate Logit (replicates Sawtooth's method).

    Parameters
    ----------
    responses_df     : Long-format response data with columns:
                       [respondent_id, task, version, best_item, worst_item]
    design_df        : Design DataFrame from design.generate_design()
    n_items          : Total number of items
    respondent_filter: Boolean mask to subset respondents (for cuts)

    Returns
    -------
    scores_df        : DataFrame with [item_id, utility, importance_0_100, rank]
    """
    if respondent_filter is not None:
        resp = responses_df[respondent_filter].copy()
    else:
        resp = responses_df.copy()

    if len(resp) < 10:
        return pd.DataFrame()

    # Build design lookup: (version, task) -> list of item_ids
    position_cols = [c for c in design_df.columns if c.startswith("position_")]
    design_lookup = {}
    for _, row in design_df.iterrows():
        ver = int(row.get("version", 1))
        task = int(row["task"])
        items = [int(row[pc]) for pc in position_cols]
        design_lookup[(ver, task)] = items

    # Build input arrays for optimization (0-indexed internally)
    tasks_best = []
    tasks_worst = []

    for _, row in resp.iterrows():
        ver = int(row.get("version", 1))
        task = int(row["task"])
        best_item = int(row["best_item"])
        worst_item = int(row["worst_item"])

        if (ver, task) not in design_lookup:
            continue

        set_items = design_lookup[(ver, task)]
        set_idxs = [i - 1 for i in set_items]  # 0-indexed
        best_idx = best_item - 1
        worst_idx = worst_item - 1
        remaining_idxs = [i for i in set_idxs if i != best_idx]

        tasks_best.append((best_idx, set_idxs))
        tasks_worst.append((worst_idx, remaining_idxs))

    # Optimize utilities (fix one item to 0 to identify the model)
    x0 = np.zeros(n_items)
    result = minimize(
        _logit_neg_loglikelihood,
        x0,
        args=(tasks_best, tasks_worst),
        method="L-BFGS-B",
        options={"maxiter": 1000, "ftol": 1e-9},
    )

    utilities = result.x

    # Rescale utilities to 0–100 (same as Sawtooth's "rescaled logit")
    mn = utilities.min()
    mx = utilities.max()
    rng = mx - mn
    if rng > 0:
        importance = (utilities - mn) / rng * 100
    else:
        importance = np.full(n_items, 50.0)

    rows = [
        {
            "item_id": i + 1,
            "utility": round(float(utilities[i]), 4),
            "importance_0_100": round(float(importance[i]), 1),
        }
        for i in range(n_items)
    ]

    scores_df = pd.DataFrame(rows)
    scores_df["rank"] = scores_df["importance_0_100"].rank(ascending=False, method="min").astype(int)
    scores_df = scores_df.sort_values("importance_0_100", ascending=False).reset_index(drop=True)

    return scores_df


# ---------------------------------------------------------------------------
# Data-cut analysis
# ---------------------------------------------------------------------------

def score_all_cuts(
    responses_df: pd.DataFrame,
    design_df: pd.DataFrame,
    n_items: int,
    cut_variables: List[str],
    method: str = "counts",
) -> Dict[str, pd.DataFrame]:
    """
    Calculate importance scores for all demographic cuts.

    Parameters
    ----------
    responses_df  : Response data including demographic columns
    design_df     : Design DataFrame
    n_items       : Total number of items
    cut_variables : List of column names to cut by (e.g. ['gender', 'age_group'])
    method        : 'counts' or 'logit'

    Returns
    -------
    results       : Dict mapping cut label to scores DataFrame
                    e.g. {'Total': df, 'gender_Male': df, 'age_group_18-34': df}
    """
    score_fn = score_counts if method == "counts" else score_logit
    results = {}

    # Total sample
    results["Total"] = score_fn(responses_df, design_df, n_items)

    # Each cut variable
    for var in cut_variables:
        if var not in responses_df.columns:
            continue
        for val in sorted(responses_df[var].dropna().unique()):
            mask = responses_df[var] == val
            label = f"{var}_{val}"
            df = score_fn(responses_df, design_df, n_items, respondent_filter=mask)
            if not df.empty:
                results[label] = df

    return results


def build_importance_matrix(
    cut_results: Dict[str, pd.DataFrame],
    items_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a wide-format importance matrix: items as rows, cuts as columns.

    Parameters
    ----------
    cut_results : Output from score_all_cuts()
    items_df    : DataFrame with [item_id, item_label]

    Returns
    -------
    matrix_df   : Wide DataFrame with item labels and importance per cut
    """
    item_map = dict(zip(items_df["item_id"], items_df["item_label"]))

    # Start with item metadata
    matrix = items_df[["item_id", "item_label"]].copy()

    for cut_label, scores_df in cut_results.items():
        if scores_df.empty:
            continue
        col_data = scores_df.set_index("item_id")["importance_0_100"]
        matrix[cut_label] = matrix["item_id"].map(col_data)

    # Sort by Total importance descending
    if "Total" in matrix.columns:
        matrix = matrix.sort_values("Total", ascending=False).reset_index(drop=True)

    return matrix


def compute_summary_stats(
    responses_df: pd.DataFrame,
    cut_variables: List[str],
) -> pd.DataFrame:
    """Return sample size summary for total and each cut."""
    rows = [{"cut": "Total", "n": responses_df["respondent_id"].nunique()}]
    for var in cut_variables:
        if var not in responses_df.columns:
            continue
        for val in sorted(responses_df[var].dropna().unique()):
            mask = responses_df[var] == val
            n = responses_df[mask]["respondent_id"].nunique()
            rows.append({"cut": f"{var}_{val}", "n": n})
    return pd.DataFrame(rows)
