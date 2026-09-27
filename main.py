"""
MaxDiff Main Script
====================
Run this script to:
  1. Generate a balanced MaxDiff card design
  2. Simulate survey responses (or load real data)
  3. Score items using Counting and/or Logit methods
  4. Export a full Excel workbook with design, scores, cuts, and simulator

Usage:
  python main.py                    # Run with defaults (15 items, 5/task, 12 tasks)
  python main.py --n_items 20       # Custom item count
  python main.py --method logit     # Use aggregate logit scoring
  python main.py --n_resp 500       # Specify respondent count
  python main.py --real_data responses.csv items.csv   # Use real survey data
"""

import argparse
import os
import sys
import pandas as pd

# Make sure the package is importable
sys.path.insert(0, os.path.dirname(__file__))

from maxdiff.design import generate_design, print_design_summary
from maxdiff.analysis import (
    score_counts, score_logit, score_all_cuts,
    build_importance_matrix, compute_summary_stats,
)
from maxdiff.excel_output import build_workbook
from maxdiff.sample_data import generate_sample_items, generate_responses


def parse_args():
    parser = argparse.ArgumentParser(
        description="MaxDiff Analysis Tool — replicates Sawtooth Software MaxDiff"
    )
    parser.add_argument("--n_items",   type=int,   default=15,
                        help="Total number of items/attributes (default: 15)")
    parser.add_argument("--k_per_task", type=int,  default=5,
                        help="Items shown per task (default: 5, Sawtooth recommends 4-5)")
    parser.add_argument("--n_tasks",   type=int,   default=12,
                        help="Tasks per respondent (default: 12)")
    parser.add_argument("--n_resp",    type=int,   default=300,
                        help="Number of respondents for simulation (default: 300)")
    parser.add_argument("--n_versions", type=int,  default=2,
                        help="Number of design versions (default: 2)")
    parser.add_argument("--method",    type=str,   default="both",
                        choices=["counts", "logit", "both"],
                        help="Scoring method: counts | logit | both (default: both)")
    parser.add_argument("--seed",      type=int,   default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--output",    type=str,   default="output/MaxDiff_Results.xlsx",
                        help="Output Excel file path")
    parser.add_argument("--real_data", nargs=2,    metavar=("RESPONSES", "ITEMS"),
                        help="Use real survey data: paths to responses.csv and items.csv")
    return parser.parse_args()


def load_real_data(responses_path: str, items_path: str):
    """
    Load real survey data from CSV files.

    responses.csv must have columns:
        respondent_id, version, task, best_item, worst_item
        + optional demographic columns (e.g. gender, age_group, region)

    items.csv must have columns:
        item_id, item_label
    """
    responses_df = pd.read_csv(responses_path)
    items_df = pd.read_csv(items_path)

    required_resp_cols = {"respondent_id", "version", "task", "best_item", "worst_item"}
    missing = required_resp_cols - set(responses_df.columns)
    if missing:
        raise ValueError(
            f"responses.csv is missing required columns: {missing}\n"
            f"Found columns: {list(responses_df.columns)}"
        )

    required_item_cols = {"item_id", "item_label"}
    missing = required_item_cols - set(items_df.columns)
    if missing:
        raise ValueError(
            f"items.csv is missing required columns: {missing}\n"
            f"Found columns: {list(items_df.columns)}"
        )

    return responses_df, items_df


def run(args):
    os.makedirs("output", exist_ok=True)

    print("\n" + "=" * 60)
    print("  MAXDIFF ANALYSIS TOOL  —  Replicating Sawtooth Software")
    print("=" * 60)

    # ---- Step 1: Items ----
    if args.real_data:
        print(f"\n[1] Loading real data from {args.real_data[0]}, {args.real_data[1]}...")
        responses_df, items_df = load_real_data(*args.real_data)
        n_items = items_df["item_id"].nunique()
        k_per_task = args.k_per_task  # user must supply this for real data
        n_tasks = args.n_tasks
    else:
        print(f"\n[1] Using sample smartphone features dataset ({args.n_items} items)...")
        items_df = generate_sample_items()
        if args.n_items != 15:
            # Trim or extend items
            items_df = items_df.head(min(args.n_items, len(items_df)))
        n_items = len(items_df)
        k_per_task = args.k_per_task
        n_tasks = args.n_tasks

    print(f"    Items: {n_items}")
    for _, row in items_df.iterrows():
        print(f"      {int(row['item_id']):2d}. {row['item_label']}")

    # ---- Step 2: Design ----
    print(f"\n[2] Generating balanced MaxDiff design...")
    print(f"    k={k_per_task} items/task, t={n_tasks} tasks, {args.n_versions} version(s)")

    design_df, design_stats = generate_design(
        n_items=n_items,
        k_per_task=k_per_task,
        n_tasks=n_tasks,
        seed=args.seed,
        n_versions=args.n_versions,
    )
    print_design_summary(design_stats)

    # ---- Step 3: Responses ----
    if args.real_data:
        print(f"\n[3] Using loaded responses: {len(responses_df):,} rows, "
              f"{responses_df['respondent_id'].nunique():,} respondents")
    else:
        print(f"\n[3] Simulating {args.n_resp:,} respondents...")
        responses_df = generate_responses(design_df, n_respondents=args.n_resp, seed=args.seed)
        print(f"    Generated {len(responses_df):,} response rows")

        # Save simulated data to CSV for reference
        design_df.to_csv("data/design_output.csv", index=False)
        responses_df.to_csv("data/sample_responses.csv", index=False)
        items_df.to_csv("data/items.csv", index=False)
        print("    Data saved to data/")

    # ---- Step 4: Identify demographic cut variables ----
    known_demo_cols = {"respondent_id", "version", "task", "best_item", "worst_item"}
    cut_variables = [c for c in responses_df.columns if c not in known_demo_cols]
    print(f"\n[4] Demographic cut variables found: {cut_variables}")

    # ---- Step 5: Score items ----
    method = args.method
    print(f"\n[5] Scoring items using method: {method.upper()}...")

    if method in ("counts", "both"):
        print("    Running Counting method...")
        cut_results_counts = score_all_cuts(
            responses_df, design_df, n_items, cut_variables, method="counts"
        )
        print(f"    ✓ Scores computed for {len(cut_results_counts)} cuts")

    if method in ("logit", "both"):
        print("    Running Aggregate Logit method...")
        cut_results_logit = score_all_cuts(
            responses_df, design_df, n_items, cut_variables, method="logit"
        )
        print(f"    ✓ Logit scores computed for {len(cut_results_logit)} cuts")

    # Choose primary results for Excel output
    if method == "logit":
        cut_results = cut_results_logit
        method_label = "Aggregate Logit"
    elif method == "counts":
        cut_results = cut_results_counts
        method_label = "Counts"
    else:
        # Use logit as primary, but show both in console
        cut_results = cut_results_logit
        method_label = "Aggregate Logit"

        # Print comparison
        print("\n    COUNTS vs LOGIT COMPARISON (Total Sample, Top 5):")
        total_c = cut_results_counts["Total"].head()
        total_l = cut_results_logit["Total"].head()
        item_map = dict(zip(items_df["item_id"], items_df["item_label"]))
        print(f"    {'Rank':<5} {'Item':<35} {'Counts':>8} {'Logit':>8}")
        print(f"    {'-'*5} {'-'*35} {'-'*8} {'-'*8}")
        for (_, rc), (_, rl) in zip(total_c.iterrows(), total_l.iterrows()):
            label = item_map.get(int(rc["item_id"]), f"Item {rc['item_id']}")
            print(f"    {int(rc['rank']):<5} {label:<35} "
                  f"{float(rc['importance_0_100']):>8.1f} "
                  f"{float(rl['importance_0_100']):>8.1f}")

    # ---- Step 6: Build importance matrix ----
    print(f"\n[6] Building importance matrix across all cuts...")
    importance_matrix = build_importance_matrix(cut_results, items_df)

    # Sample size per cut
    n_summary = compute_summary_stats(responses_df, cut_variables)
    n_by_cut = dict(zip(n_summary["cut"], n_summary["n"]))
    # Add item_id and item_label keys with 0 (they're not cuts)
    n_by_cut["item_id"] = ""
    n_by_cut["item_label"] = ""

    # ---- Step 7: Build Excel workbook ----
    print(f"\n[7] Building Excel workbook: {args.output}")
    build_workbook(
        output_path=args.output,
        design_df=design_df,
        design_stats=design_stats,
        responses_df=responses_df,
        items_df=items_df,
        cut_results=cut_results,
        importance_matrix=importance_matrix,
        cut_variables=cut_variables,
        n_respondents_by_cut=n_by_cut,
        method_label=method_label,
    )

    print("\n" + "=" * 60)
    print("  COMPLETE!")
    print(f"  Output: {os.path.abspath(args.output)}")
    print()
    print("  Sheets in workbook:")
    print("    HOW_IT_WORKS     — Methodology & design explanation")
    print("    Design           — Card layout matrix (tasks × items)")
    print("    Raw_Data         — Survey responses in long format")
    print("    Importance_Total — Overall scores with bar chart")
    print("    Importance_Cuts  — Scores by demographic cuts")
    print("    Simulator        — Interactive subgroup comparison")
    print()
    print("  To use the Simulator:")
    print("    1. Open the 'Simulator' sheet")
    print("    2. Click the selector cell (C5)")
    print("    3. Choose a subgroup from the dropdown")
    print("    4. The table and chart update automatically")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    args = parse_args()
    run(args)
