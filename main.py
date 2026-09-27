"""
MaxDiff Main Script
====================
Accepts a list of attributes (items) and sample size, then:
  1. Auto-selects optimal card design parameters
  2. Checks whether the design can be fully orthogonal (BIBD) or near-orthogonal
  3. Generates the balanced card design
  4. Scores items (Counting and/or Aggregate Logit)
  5. Exports a full Excel workbook

INPUT FORMATS
--------------
Option A — CSV file:
    attributes.csv:
        item_id,item_label
        1,Battery life
        2,Camera quality
        ...

Option B — inline list (for quick demos):
    python main.py --demo_attrs "Battery life,Camera quality,Price,Durability"

Option C — legacy numeric mode:
    python main.py --n_items 15 --k_per_task 5 --n_tasks 12

USAGE
------
  python main.py --attributes data/items.csv --n_resp 300
  python main.py --attributes data/items.csv --n_resp 500 --method logit
  python main.py --demo_attrs "Feature A,Feature B,Feature C,Feature D,Feature E" --n_resp 200
  python main.py --real_data data/responses.csv data/items.csv
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
from typing import Dict

sys.path.insert(0, os.path.dirname(__file__))

from maxdiff.design import (
    generate_design_from_attributes, print_design_summary,
    check_bibd_feasibility, find_best_bibd, auto_design_params,
)
from maxdiff.analysis import (
    score_counts, score_logit, score_all_cuts,
    build_importance_matrix, compute_summary_stats,
)
from maxdiff.excel_output import build_workbook
from maxdiff.sample_data import generate_responses
from maxdiff.input_parser import parse_study_setup, find_default_setup

_INIT_TEMPLATE = """\
# ============================================================
#  MaxDiff Study Configuration
#  Fill in the sections below, then run:
#    python main.py
# ============================================================

# --- Study settings -----------------------------------------
STUDY_NAME    = My MaxDiff Study
N_RESPONDENTS = 300

# Optional overrides (auto-selected when left blank)
# K_PER_TASK  = 5     # items shown per card (Sawtooth recommends 4–5)
# N_TASKS     = 12    # cards per respondent
# N_VERSIONS  = 2     # number of design versions
# METHOD      = both  # counts | logit | both
# OUTPUT      = output/MaxDiff_Results.xlsx

# --- Attributes ---------------------------------------------
# One attribute per line (no commas needed).
# Lines starting with # are ignored.
ATTRIBUTES:
Attribute 1
Attribute 2
Attribute 3
Attribute 4
Attribute 5
Attribute 6
Attribute 7
Attribute 8
"""


def parse_args():
    p = argparse.ArgumentParser(
        description="MaxDiff Analysis — Sawtooth-style card design, scoring & Excel output"
    )

    # --- Setup file (primary input) ---
    p.add_argument("--setup", type=str, default=None,
                   help="Path to study_setup.txt (default: input/study_setup.txt if found).")
    p.add_argument("--init", action="store_true",
                   help="Create a blank input/study_setup.txt template and exit.")

    # --- Attribute overrides (alternative to setup file) ---
    p.add_argument("--attributes", type=str, default=None,
                   help="Path to items CSV (columns: item_id, item_label).")
    p.add_argument("--demo_attrs", type=str, default=None,
                   help="Comma-separated inline attribute list for quick demos.")
    p.add_argument("--n_resp", type=int, default=None,
                   help="Number of respondents (overrides setup file; default: 300).")

    # --- Design overrides (optional — auto-selected when omitted) ---
    p.add_argument("--k_per_task", type=int, default=None,
                   help="Items per card (auto-selected if omitted; Sawtooth recommends 4–5).")
    p.add_argument("--n_tasks", type=int, default=None,
                   help="Cards per respondent (auto-selected if omitted).")
    p.add_argument("--n_versions", type=int, default=None,
                   help="Number of design versions (default: 2).")

    # --- Analysis ---
    p.add_argument("--method", type=str, default=None,
                   choices=["counts", "logit", "both"],
                   help="Scoring method: counts | logit | both (default: both).")

    # --- Real data mode ---
    p.add_argument("--real_data", nargs=2, metavar=("RESPONSES_CSV", "ITEMS_CSV"),
                   help="Use real survey data instead of simulation.")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=str, default=None,
                   help="Output Excel path (default: output/MaxDiff_Results.xlsx).")
    return p.parse_args()


def _init_template():
    """Write a blank study_setup.txt template and exit."""
    os.makedirs("input", exist_ok=True)
    dest = os.path.join("input", "study_setup.txt")
    if os.path.exists(dest):
        print(f"  {dest} already exists — not overwriting.")
    else:
        with open(dest, "w", encoding="utf-8") as fh:
            fh.write(_INIT_TEMPLATE)
        print(f"  Created {dest}")
        print("  Edit it to add your study name, respondent count, and attributes.")
        print("  Then run:  python main.py")
    sys.exit(0)


def _merge_setup_with_args(args):
    """
    Resolve final run parameters by merging setup file (if present) with
    CLI overrides.  Returns a namespace-like object with the same attributes
    as `args` plus any values loaded from the setup file.
    """
    # Determine setup file path
    setup_path = args.setup or find_default_setup()

    if setup_path and not args.attributes and not args.demo_attrs:
        cfg = parse_study_setup(setup_path)
        print(f"    Using setup file: {setup_path}")
        # CLI flags take precedence over file settings
        args.n_resp      = args.n_resp      or cfg["n_respondents"]
        args.k_per_task  = args.k_per_task  or cfg["k_per_task"]
        args.n_tasks     = args.n_tasks     or cfg["n_tasks"]
        args.n_versions  = args.n_versions  or cfg["n_versions"]
        args.method      = args.method      or cfg["method"]
        args.output      = args.output      or cfg["output"]
        args._items_df   = cfg["items_df"]
    else:
        args._items_df   = None
        args.n_resp      = args.n_resp      or 300
        args.n_versions  = args.n_versions  or 2
        args.method      = args.method      or "both"
        args.output      = args.output      or "output/MaxDiff_Results.xlsx"

    return args


def load_attributes(args) -> pd.DataFrame:
    """Load attributes from setup file, CSV, inline string, or built-in sample."""
    if getattr(args, "_items_df", None) is not None:
        return args._items_df

    if args.demo_attrs:
        labels = [x.strip() for x in args.demo_attrs.split(",") if x.strip()]
        return pd.DataFrame({
            "item_id": list(range(1, len(labels) + 1)),
            "item_label": labels,
        })

    if args.attributes:
        df = pd.read_csv(args.attributes)
        if "item_id" not in df.columns or "item_label" not in df.columns:
            raise ValueError("attributes CSV must have columns: item_id, item_label")
        return df[["item_id", "item_label"]].copy()

    # Default: built-in smartphone features
    from maxdiff.sample_data import generate_sample_items
    return generate_sample_items()


def print_orthogonality_explanation(stats: Dict, n_resp: int) -> None:
    """Print a clear explanation of the orthogonality result."""
    feas = stats.get("feasibility", {})
    is_orth = stats.get("is_orthogonal", False)
    n_items = stats["n_items"]
    k = stats["k_per_task"]

    print()
    print("  ORTHOGONALITY CHECK")
    print("  " + "-" * 60)
    print(f"  N attributes = {n_items},  k per card = {k}")
    print()

    from maxdiff.design import check_bibd_feasibility as _cbibd, _KNOWN_DIFF_SETS

    # Show BIBD check for λ = 1, 2, 3
    print("  Checking BIBD conditions (λ = times each pair co-occurs):")
    print(f"  {'λ':>4}  {'r (appear.)':>12}  {'b (tasks)':>10}  {'Feasible?':>10}")
    print(f"  {'─'*4}  {'─'*12}  {'─'*10}  {'─'*10}")
    for lam in range(1, 4):
        info = _cbibd(n_items, k, lam)
        r_str = f"{info['r_appearances']:.1f}" if not info['r_is_integer'] else str(int(info['r_appearances']))
        b_str = f"{info['b_tasks']:.1f}" if not info['b_is_integer'] else str(int(info['b_tasks']))
        ok = "✓ YES" if info['bibd_necessary_conditions_met'] else "✗ NO"
        print(f"  {lam:>4}  {r_str:>12}  {b_str:>10}  {ok:>10}")
    print()

    # Check if BIBD conditions are met numerically (even if no construction available)
    bibd_met_no_construct = False
    for lam in range(1, 4):
        info = _cbibd(n_items, k, lam)
        if (info["bibd_necessary_conditions_met"]
                and info["n_tasks_needed"] is not None
                and info["n_tasks_needed"] <= 24
                and (n_items, k, lam) not in _KNOWN_DIFF_SETS):
            bibd_met_no_construct = True
            bibd_no_construct_info = info
            break

    if is_orth:
        lam = feas.get("lambda", "?")
        b = feas.get("n_tasks_needed", "?")
        print(f"  ✓  ORTHOGONAL DESIGN achieved with λ={lam}")
        print(f"     Every pair of attributes co-occurs exactly {lam} time(s) per respondent.")
        print(f"     This required b={b} tasks (all shown to each respondent).")
    elif bibd_met_no_construct:
        lam = bibd_no_construct_info.get("lambda", "?")
        b = bibd_no_construct_info.get("n_tasks_needed", "?")
        print(f"  ⚠  BIBD necessary conditions ARE met (λ={lam}, b={b} tasks),")
        print(f"     but no explicit construction is implemented for N={n_items}, k={k}.")
        print(f"     Falling back to NEAR-ORTHOGONAL (balanced) design.")
    else:
        print(f"  ✗  True BIBD not achievable within ≤24 tasks for these parameters.")
        print(f"     Using NEAR-ORTHOGONAL (balanced) design instead.")
        mn = stats.get("pair_counts_min", "?")
        mx = stats.get("pair_counts_max", "?")
        cv = stats.get("pair_cv", "?")
        mean_app = stats.get("mean_appearances", "?")
        print(f"     Each attribute appears {mean_app:.1f} times per respondent (balanced).")
        print(f"     Pair co-occurrence: {mn}–{mx} times (CV={cv:.3f}; 0 = perfect).")
        print()
        print("  NOTE: Near-orthogonal designs are the Sawtooth standard when BIBD")
        print("  is infeasible. Scores correlate r>0.95 with true BIBD results.")
    print()


def run(args):
    if args.init:
        _init_template()

    args = _merge_setup_with_args(args)

    os.makedirs("output", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    print("\n" + "=" * 65)
    print("  MAXDIFF ANALYSIS TOOL  —  Replicating Sawtooth Software")
    print("=" * 65)

    # ── Step 1: Load attributes ──────────────────────────────────────────
    print(f"\n[1] Loading attributes...")
    items_df = load_attributes(args)
    n_items = len(items_df)

    print(f"    {n_items} attributes loaded:")
    for _, row in items_df.iterrows():
        print(f"      {int(row['item_id']):2d}. {row['item_label']}")

    # ── Step 2: Auto-select design parameters ────────────────────────────
    print(f"\n[2] Selecting optimal design parameters for N={n_items}, n={args.n_resp}...")
    params = auto_design_params(n_items, args.n_resp)
    k_final = args.k_per_task or params["k_per_task"]
    t_final = args.n_tasks or params["n_tasks"]

    print(f"    k per card       : {k_final}")
    print(f"    t cards/respondent: {t_final}")
    print(f"    Design type      : {params['design_type']}")
    print(f"    Orthogonal (BIBD): {'YES' if params['is_orthogonal'] else 'NO — near-orthogonal'}")

    # ── Step 3: Generate design ──────────────────────────────────────────
    print(f"\n[3] Generating card design ({args.n_versions} version(s))...")
    attr_list = items_df["item_label"].tolist()

    items_df, design_df, stats = generate_design_from_attributes(
        attributes=attr_list,
        n_respondents=args.n_resp,
        k_per_task=k_final,
        n_tasks=t_final,
        seed=args.seed,
        n_versions=args.n_versions,
    )
    print_design_summary(stats)
    print_orthogonality_explanation(stats, args.n_resp)

    # ── Step 4: Survey responses ─────────────────────────────────────────
    if args.real_data:
        print(f"\n[4] Loading real survey data: {args.real_data[0]}")
        responses_df = pd.read_csv(args.real_data[0])
        req = {"respondent_id", "version", "task", "best_item", "worst_item"}
        missing = req - set(responses_df.columns)
        if missing:
            raise ValueError(f"responses CSV missing columns: {missing}")
        print(f"    {len(responses_df):,} rows, {responses_df['respondent_id'].nunique():,} respondents")
    else:
        print(f"\n[4] Simulating {args.n_resp:,} respondents...")
        responses_df = generate_responses(design_df, n_respondents=args.n_resp, seed=args.seed)
        print(f"    Generated {len(responses_df):,} response rows")
        design_df.to_csv("data/design_output.csv", index=False)
        responses_df.to_csv("data/sample_responses.csv", index=False)
        items_df.to_csv("data/items.csv", index=False)
        print("    Data saved to data/")

    # ── Step 5: Cut variables ────────────────────────────────────────────
    known_cols = {"respondent_id", "version", "task", "best_item", "worst_item"}
    cut_variables = [c for c in responses_df.columns if c not in known_cols]
    if cut_variables:
        print(f"\n[5] Cut variables: {cut_variables}")
    else:
        print(f"\n[5] No demographic cut variables found (Total only)")

    # ── Step 6: Score items ──────────────────────────────────────────────
    method = args.method
    print(f"\n[6] Scoring ({method.upper()})...")
    cut_results_counts = cut_results_logit = None

    if method in ("counts", "both"):
        cut_results_counts = score_all_cuts(
            responses_df, design_df, n_items, cut_variables, method="counts"
        )
        print(f"    ✓ Counting scores: {len(cut_results_counts)} cuts")

    if method in ("logit", "both"):
        cut_results_logit = score_all_cuts(
            responses_df, design_df, n_items, cut_variables, method="logit"
        )
        print(f"    ✓ Logit scores: {len(cut_results_logit)} cuts")

    # Print comparison if both methods run
    if method == "both":
        item_map = dict(zip(items_df["item_id"], items_df["item_label"]))
        total_c = cut_results_counts["Total"].head(5)
        total_l = cut_results_logit["Total"].head(5)
        print()
        print(f"    {'Rank':<5} {'Attribute':<35} {'Counts':>8} {'Logit':>8}")
        print(f"    {'-'*5} {'-'*35} {'-'*8} {'-'*8}")
        for (_, rc), (_, rl) in zip(total_c.iterrows(), total_l.iterrows()):
            label = item_map.get(int(rc["item_id"]), f"Item {rc['item_id']}")[:34]
            print(f"    {int(rc['rank']):<5} {label:<35} "
                  f"{float(rc['importance_0_100']):>8.1f} "
                  f"{float(rl['importance_0_100']):>8.1f}")

    # Primary results for Excel
    if method == "counts":
        cut_results, method_label = cut_results_counts, "Counts"
    elif method == "logit":
        cut_results, method_label = cut_results_logit, "Aggregate Logit"
    else:
        cut_results, method_label = cut_results_logit, "Aggregate Logit"

    # ── Step 7: Importance matrix ─────────────────────────────────────────
    print(f"\n[7] Building importance matrix...")
    importance_matrix = build_importance_matrix(cut_results, items_df)
    n_summary = compute_summary_stats(responses_df, cut_variables)
    n_by_cut = dict(zip(n_summary["cut"], n_summary["n"]))
    n_by_cut.update({"item_id": "", "item_label": ""})

    # ── Step 8: Export Excel ──────────────────────────────────────────────
    print(f"\n[8] Building Excel workbook → {args.output}")
    build_workbook(
        output_path=args.output,
        design_df=design_df,
        design_stats=stats,
        responses_df=responses_df,
        items_df=items_df,
        cut_results=cut_results,
        importance_matrix=importance_matrix,
        cut_variables=cut_variables,
        n_respondents_by_cut=n_by_cut,
        method_label=method_label,
    )

    print("\n" + "=" * 65)
    print("  COMPLETE")
    print(f"  Output : {os.path.abspath(args.output)}")
    print()
    print("  Workbook sheets:")
    print("    HOW_IT_WORKS     — Methodology, design logic, orthogonality")
    print("    Design           — Card layout (tasks × attributes)")
    print("    Raw_Data         — Survey responses")
    print("    Importance_Total — Overall scores + bar chart")
    print("    Importance_Cuts  — Heat-map by demographic cuts")
    print("    Simulator        — Interactive subgroup comparator (dropdown)")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    args = parse_args()
    run(args)
