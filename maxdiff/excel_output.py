"""
MaxDiff Excel Workbook Builder
================================
Creates a multi-sheet Excel workbook replicating Sawtooth Software's output:

Sheets produced:
  1. HOW_IT_WORKS   — Methodology explanation (design logic, scoring, interpretation)
  2. Design         — Card design matrix (which items appear in each task)
  3. Raw_Data       — Raw survey responses in long format
  4. Importance_Total — Overall importance scores with bar chart
  5. Importance_Cuts  — Scores broken out by each demographic variable
  6. Simulator      — Interactive importance simulator with dropdown selectors
                       (select a subgroup → scores and chart update automatically
                        via Excel VLOOKUP formulas, no VBA required)
"""

import pandas as pd
import numpy as np
import openpyxl
from openpyxl import Workbook
from openpyxl.styles import (
    PatternFill, Font, Alignment, Border, Side, GradientFill
)
from openpyxl.chart import BarChart, Reference
from openpyxl.chart.series import DataPoint
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation
from typing import Dict, List, Optional
import io


# ---------------------------------------------------------------------------
# Colour palette (matching a professional market research look)
# ---------------------------------------------------------------------------
COLOURS = {
    "header_dark":   "1F3864",   # deep navy
    "header_mid":    "2E5E9E",   # medium blue
    "header_light":  "C5D6E8",   # pale blue
    "accent_orange": "E67E22",   # highlight
    "accent_green":  "27AE60",   # positive
    "accent_red":    "E74C3C",   # negative / caution
    "row_alt":       "EBF3FB",   # alternating row
    "white":         "FFFFFF",
    "light_grey":    "F2F2F2",
    "mid_grey":      "BFBFBF",
    "dark_text":     "1A1A2E",
    "bar_blue":      "2E75B6",
    "bar_orange":    "ED7D31",
    "simulator_bg":  "FFF9E6",   # warm cream for simulator
    "sim_header":    "5C3317",   # dark brown for simulator header
}

# Bar colours cycled for multi-cut charts
BAR_COLOURS = ["2E75B6", "ED7D31", "70AD47", "FFC000", "9E480E", "843C0C"]


def _fill(hex_colour: str) -> PatternFill:
    return PatternFill("solid", fgColor=hex_colour)


def _font(bold=False, size=11, colour="1A1A2E", italic=False) -> Font:
    return Font(bold=bold, size=size, color=colour, italic=italic, name="Calibri")


def _border(style="thin") -> Border:
    s = Side(style=style, color="BFBFBF")
    return Border(left=s, right=s, top=s, bottom=s)


def _header_style(ws, row, col_start, col_end, text, bg="header_dark",
                  fg="FFFFFF", size=12, merge=True):
    c = ws.cell(row=row, column=col_start, value=text)
    c.fill = _fill(COLOURS[bg])
    c.font = _font(bold=True, size=size, colour=fg)
    c.alignment = Alignment(horizontal="center", vertical="center")
    if merge and col_end > col_start:
        ws.merge_cells(
            start_row=row, start_column=col_start,
            end_row=row, end_column=col_end
        )
    return c


def _auto_width(ws, extra=4):
    """Auto-size column widths."""
    for col in ws.columns:
        max_len = 0
        col_letter = get_column_letter(col[0].column)
        for cell in col:
            try:
                if cell.value:
                    max_len = max(max_len, len(str(cell.value)))
            except Exception:
                pass
        ws.column_dimensions[col_letter].width = min(max_len + extra, 50)


# ---------------------------------------------------------------------------
# Sheet 1: How It Works (methodology explanation)
# ---------------------------------------------------------------------------

def _write_how_it_works(ws, n_items, k_per_task, n_tasks, design_stats):
    ws.sheet_view.showGridLines = False
    ws.column_dimensions["A"].width = 2
    ws.column_dimensions["B"].width = 28
    ws.column_dimensions["C"].width = 72

    sections = [
        ("WHAT IS MAXDIFF?",
         "MaxDiff (Maximum Difference Scaling), also called Best-Worst Scaling, is a "
         "conjoint-style survey method developed by Jordan Louviere in the 1990s and "
         "commercialised by Sawtooth Software.\n\n"
         "Respondents are shown small sets of items (attributes, features, or concepts) "
         "and asked to pick the BEST (most preferred / most important) and the WORST "
         "(least preferred / least important) from each set.\n\n"
         "Repeating this across many sets produces rich ranking data that reveals the "
         "relative importance of all items — not just simple top-box ratings."),

        ("WHY MAXDIFF OVER RATINGS?",
         "Rating scales (1–10, Top-2-Box) suffer from:\n"
         "  • Acquiescence bias — respondents give high ratings to everything\n"
         "  • Scale usage differences across cultures and individuals\n"
         "  • Inflation: 70% of items rated 8+ tell you nothing about relative priority\n\n"
         "MaxDiff forces trade-offs — you cannot say everything is equally important — "
         "producing discrimination that ratings cannot."),

        ("CARD DESIGN PARAMETERS",
         f"This study uses:\n"
         f"  • N = {n_items} total items\n"
         f"  • k = {k_per_task} items shown per task (Sawtooth recommends 4 or 5)\n"
         f"  • t = {n_tasks} tasks per respondent\n"
         f"  • Appearances per item ≈ {design_stats['feasibility']['appearances_per_item']:.1f}\n\n"
         f"Balance rule: Each item must appear in at least 2 tasks, ideally 3+.\n"
         f"Current design: min={design_stats['min_appearances']}, "
         f"max={design_stats['max_appearances']}, "
         f"mean={design_stats['mean_appearances']:.2f}"),

        ("HOW THE DESIGN IS GENERATED",
         "The design uses a GREEDY ROTATION ALGORITHM (similar to Sawtooth's approach):\n\n"
         "Step 1: Initialise an appearance counter for each of the N items.\n"
         "Step 2: For each task, sort items by fewest appearances so far.\n"
         "Step 3: Select the k items with the lowest appearance counts.\n"
         "         (Ties are broken randomly to ensure position balance.)\n"
         "Step 4: Randomise the within-task item order.\n"
         "         (This prevents position-order bias in best/worst choices.)\n"
         "Step 5: Update the appearance counter and repeat.\n\n"
         "Goal: Each item appears in roughly t×k/N tasks, creating a\n"
         "Balanced Incomplete Block Design (BIBD) or near-BIBD."),

        ("SCORING: COUNTING METHOD",
         "The simplest scoring approach (Sawtooth 'Counts'):\n\n"
         "  best_rate(i)  = times item i chosen as BEST / times item i appeared\n"
         "  worst_rate(i) = times item i chosen as WORST / times item i appeared\n"
         "  raw_score(i)  = best_rate(i) − worst_rate(i)   [range: −1 to +1]\n\n"
         "Rescale to 0–100:\n"
         "  importance(i) = (raw_score(i) − min) / (max − min) × 100\n\n"
         "The least important item scores 0; the most important scores 100.\n"
         "This correlates r > 0.95 with logit scores for most designs."),

        ("SCORING: AGGREGATE LOGIT METHOD",
         "The more precise scoring method (matches Sawtooth's aggregate logit):\n\n"
         "Assumes a Random Utility Model:\n"
         "  P(best=i | set S) = exp(u_i) / Σⱼ exp(u_j)   for j in S\n"
         "  P(worst=i | set S) = exp(−u_i) / Σⱼ exp(−u_j)  for j in S\\{best}\n\n"
         "Utilities u_i are estimated via Maximum Likelihood Estimation (MLE)\n"
         "using the L-BFGS-B optimiser over all best AND worst choices simultaneously.\n\n"
         "Output utilities are on an interval scale: differences are meaningful,\n"
         "but there is no natural zero point. They are then rescaled to 0–100."),

        ("INTERPRETING THE SIMULATOR",
         "The Simulator sheet lets you compare importance scores across subgroups:\n\n"
         "1. Select a Cut Variable from the dropdown (e.g. 'Gender')\n"
         "2. The chart and table update automatically via Excel VLOOKUP formulas\n"
         "3. No macros or VBA are needed — Excel recalculates on selection change\n\n"
         "Use cases:\n"
         "  • Is Battery Life more important to younger vs older respondents?\n"
         "  • Do males and females differ in their Camera Quality priority?\n"
         "  • Which features should we emphasise in each regional market?"),
    ]

    row = 2
    _header_style(ws, row, 2, 3, "MAXDIFF METHODOLOGY & DESIGN GUIDE",
                  bg="header_dark", size=14)
    ws.row_dimensions[row].height = 30
    row += 2

    for title, content in sections:
        # Section header
        c = ws.cell(row=row, column=2, value=title)
        c.fill = _fill(COLOURS["header_mid"])
        c.font = _font(bold=True, size=11, colour="FFFFFF")
        c.alignment = Alignment(horizontal="left", vertical="center", wrap_text=False)
        c.border = _border()
        ws.merge_cells(start_row=row, start_column=2, end_row=row, end_column=3)
        ws.row_dimensions[row].height = 22
        row += 1

        # Content
        c = ws.cell(row=row, column=2)
        c2 = ws.cell(row=row, column=3, value=content)
        c2.font = _font(size=10)
        c2.alignment = Alignment(horizontal="left", vertical="top", wrap_text=True)
        c2.fill = _fill(COLOURS["light_grey"])
        ws.merge_cells(start_row=row, start_column=2, end_row=row, end_column=3)
        # Approximate row height based on content lines
        lines = content.count("\n") + 3
        ws.row_dimensions[row].height = max(15 * lines, 30)
        row += 2


# ---------------------------------------------------------------------------
# Sheet 2: Design Matrix
# ---------------------------------------------------------------------------

def _write_design(ws, design_df, items_df, stats):
    ws.sheet_view.showGridLines = False
    item_map = dict(zip(items_df["item_id"], items_df["item_label"]))
    k = len([c for c in design_df.columns if c.startswith("position_")])

    versions = sorted(design_df["version"].unique())
    row = 1

    for ver in versions:
        v_design = design_df[design_df["version"] == ver]

        # Version header
        _header_style(ws, row, 1, k + 2,
                      f"MAXDIFF DESIGN — VERSION {ver}  "
                      f"({len(v_design)} tasks × {k} items per task, "
                      f"{stats['n_items']} total items)",
                      bg="header_dark", size=12)
        ws.row_dimensions[row].height = 28
        row += 1

        # Appearances summary row
        ws.cell(row=row, column=1, value="Design balance:")
        ws.cell(row=row, column=2,
                value=f"Min appearances/item = {stats['min_appearances']} | "
                      f"Max = {stats['max_appearances']} | "
                      f"Mean = {stats['mean_appearances']:.2f} | "
                      f"Balance ratio = {stats.get('item_balance_ratio', stats.get('balance_ratio', 0)):.3f}")
        for col in range(1, k + 3):
            ws.cell(row=row, column=col).fill = _fill(COLOURS["light_grey"])
            ws.cell(row=row, column=col).font = _font(size=9, italic=True)
        ws.merge_cells(start_row=row, start_column=2, end_row=row, end_column=k + 2)
        row += 1

        # Column headers
        headers = ["Task"] + [f"Position {p}" for p in range(1, k + 1)] + ["Item IDs"]
        for col, h in enumerate(headers, 1):
            c = ws.cell(row=row, column=col, value=h)
            c.fill = _fill(COLOURS["header_light"])
            c.font = _font(bold=True, size=10)
            c.alignment = Alignment(horizontal="center")
            c.border = _border()
        row += 1

        # Data rows
        for i, (_, drow) in enumerate(v_design.iterrows()):
            bg = COLOURS["white"] if i % 2 == 0 else COLOURS["row_alt"]
            ws.cell(row=row, column=1, value=int(drow["task"])).fill = _fill(bg)
            ws.cell(row=row, column=1).font = _font(bold=True, size=10)
            ws.cell(row=row, column=1).alignment = Alignment(horizontal="center")

            item_ids = []
            for pos in range(1, k + 1):
                item_id = int(drow[f"position_{pos}"])
                label = item_map.get(item_id, f"Item {item_id}")
                c = ws.cell(row=row, column=pos + 1, value=label)
                c.fill = _fill(bg)
                c.font = _font(size=10)
                c.alignment = Alignment(horizontal="left")
                c.border = _border()
                item_ids.append(str(item_id))

            id_cell = ws.cell(row=row, column=k + 2, value=", ".join(item_ids))
            id_cell.fill = _fill(bg)
            id_cell.font = _font(size=9, italic=True, colour="808080")
            row += 1

        row += 2  # gap between versions

    # Column widths
    ws.column_dimensions["A"].width = 8
    for pos in range(1, k + 1):
        ws.column_dimensions[get_column_letter(pos + 1)].width = 28
    ws.column_dimensions[get_column_letter(k + 2)].width = 20


# ---------------------------------------------------------------------------
# Sheet 3: Raw Data
# ---------------------------------------------------------------------------

def _write_raw_data(ws, responses_df, items_df):
    ws.sheet_view.showGridLines = False
    item_map = dict(zip(items_df["item_id"], items_df["item_label"]))

    # Add label columns
    data = responses_df.copy()
    data["best_label"] = data["best_item"].map(item_map)
    data["worst_label"] = data["worst_item"].map(item_map)

    cols = list(data.columns)
    _header_style(ws, 1, 1, len(cols), "RAW SURVEY RESPONSES",
                  bg="header_dark", size=12)
    ws.row_dimensions[1].height = 24

    for col, header in enumerate(cols, 1):
        c = ws.cell(row=2, column=col, value=header.upper().replace("_", " "))
        c.fill = _fill(COLOURS["header_mid"])
        c.font = _font(bold=True, size=10, colour="FFFFFF")
        c.alignment = Alignment(horizontal="center")
        c.border = _border()

    for r_idx, (_, row) in enumerate(data.iterrows(), 3):
        bg = COLOURS["white"] if r_idx % 2 == 0 else COLOURS["row_alt"]
        for col, val in enumerate(row, 1):
            c = ws.cell(row=r_idx, column=col, value=val)
            c.fill = _fill(bg)
            c.font = _font(size=9)
            c.alignment = Alignment(horizontal="center")
            c.border = _border()

    _auto_width(ws)


# ---------------------------------------------------------------------------
# Sheet 4: Importance Total
# ---------------------------------------------------------------------------

def _write_importance_total(ws, scores_df, items_df, n_respondents, method_label):
    ws.sheet_view.showGridLines = False
    item_map = dict(zip(items_df["item_id"], items_df["item_label"]))

    scores = scores_df.copy()
    scores["item_label"] = scores["item_id"].map(item_map)
    scores = scores.sort_values("importance_0_100", ascending=False).reset_index(drop=True)

    n = len(scores)

    # Title
    _header_style(ws, 1, 1, 7,
                  f"MAXDIFF IMPORTANCE SCORES — TOTAL SAMPLE  "
                  f"(n={n_respondents:,} respondents, method={method_label})",
                  bg="header_dark", size=12)
    ws.row_dimensions[1].height = 28

    # Sub-header
    _header_style(ws, 2, 1, 7,
                  "Scores rescaled 0–100. Highest item = 100 (most important), "
                  "lowest item = 0 (least important). Differences are relative, not absolute.",
                  bg="header_light", fg="1A1A2E", size=9, merge=True)
    ws.row_dimensions[2].height = 18

    # Column headers
    col_headers = ["Rank", "Item", "Importance (0–100)", "Best %", "Worst %",
                   "Raw Score", "Appearances"]
    for col, h in enumerate(col_headers, 1):
        c = ws.cell(row=3, column=col, value=h)
        c.fill = _fill(COLOURS["header_mid"])
        c.font = _font(bold=True, size=10, colour="FFFFFF")
        c.alignment = Alignment(horizontal="center")
        c.border = _border()
    ws.row_dimensions[3].height = 20

    # Data rows
    for i, (_, row) in enumerate(scores.iterrows()):
        r = i + 4
        bg = COLOURS["white"] if i % 2 == 0 else COLOURS["row_alt"]

        vals = [
            int(row["rank"]),
            row["item_label"],
            round(float(row["importance_0_100"]), 1),
            round(float(row.get("best_rate", 0)) * 100, 1),
            round(float(row.get("worst_rate", 0)) * 100, 1),
            round(float(row.get("raw_score", row.get("utility", 0))), 4),
            int(row.get("appearances", 0)),
        ]
        aligns = ["center", "left", "center", "center", "center", "center", "center"]

        for col, (val, align) in enumerate(zip(vals, aligns), 1):
            c = ws.cell(row=r, column=col, value=val)
            c.fill = _fill(bg)
            c.font = _font(size=10, bold=(col == 1))
            c.alignment = Alignment(horizontal=align)
            c.border = _border()

        # Colour-code importance bar visually
        imp_cell = ws.cell(row=r, column=3)
        imp_val = float(row["importance_0_100"])
        if imp_val >= 70:
            imp_cell.fill = _fill("C6EFCE")   # green
        elif imp_val >= 40:
            imp_cell.fill = _fill("FFEB9C")   # yellow
        else:
            imp_cell.fill = _fill("FFC7CE")   # red

    # Bar chart
    chart_start_row = 4
    items_ref = Reference(ws, min_col=2, min_row=chart_start_row,
                           max_row=chart_start_row + n - 1)
    data_ref = Reference(ws, min_col=3, min_row=chart_start_row - 0,
                          max_row=chart_start_row + n - 1)

    chart = BarChart()
    chart.type = "bar"
    chart.grouping = "clustered"
    chart.title = "MaxDiff Importance Scores — Total Sample"
    chart.y_axis.title = "Item"
    chart.x_axis.title = "Importance (0–100)"
    chart.add_data(data_ref)
    chart.set_categories(items_ref)
    chart.series[0].graphicalProperties.solidFill = COLOURS["bar_blue"]
    chart.series[0].graphicalProperties.line.solidFill = COLOURS["bar_blue"]
    chart.shape = 4
    chart.width = 20
    chart.height = max(10, n * 0.6)
    ws.add_chart(chart, f"I3")

    # Column widths
    ws.column_dimensions["A"].width = 7
    ws.column_dimensions["B"].width = 32
    for col_letter in ["C", "D", "E", "F", "G"]:
        ws.column_dimensions[col_letter].width = 18


# ---------------------------------------------------------------------------
# Sheet 5: Importance by Cuts
# ---------------------------------------------------------------------------

def _write_importance_cuts(ws, importance_matrix, items_df, cut_variables, n_respondents_by_cut):
    ws.sheet_view.showGridLines = False

    cols = list(importance_matrix.columns)
    n_items = len(importance_matrix)

    _header_style(ws, 1, 1, len(cols),
                  "MAXDIFF IMPORTANCE SCORES BY DEMOGRAPHIC CUTS (0–100 Scale)",
                  bg="header_dark", size=12)
    ws.row_dimensions[1].height = 28

    # Header row with cut labels and sample sizes
    for col, header in enumerate(cols, 1):
        n = n_respondents_by_cut.get(header, "")
        display = header.replace("_", " ").title()
        if n:
            display += f"\n(n={n:,})"
        c = ws.cell(row=2, column=col, value=display)
        c.fill = _fill(COLOURS["header_mid"])
        c.font = _font(bold=True, size=10, colour="FFFFFF")
        c.alignment = Alignment(horizontal="center", wrap_text=True)
        c.border = _border()
    ws.row_dimensions[2].height = 30

    for r_idx, (_, row) in enumerate(importance_matrix.iterrows()):
        r = r_idx + 3
        bg = COLOURS["white"] if r_idx % 2 == 0 else COLOURS["row_alt"]
        for col, val in enumerate(row, 1):
            c = ws.cell(row=r, column=col, value=val)
            c.fill = _fill(bg)
            c.alignment = Alignment(horizontal="center" if col > 2 else "left")
            c.border = _border()
            if col == 1:
                c.font = _font(bold=True, size=10)
            elif col == 2:
                c.font = _font(size=10)
            else:
                # Heat-map colour based on value
                try:
                    v = float(val)
                    if v >= 70:
                        c.fill = _fill("C6EFCE")
                    elif v >= 40:
                        c.fill = _fill("FFEB9C")
                    else:
                        c.fill = _fill("FFC7CE")
                    c.font = _font(size=10, bold=(col == 3))  # bold for Total
                except Exception:
                    pass

    # Add a multi-series bar chart for Total vs first cut variable
    if len(cols) > 3:
        items_ref = Reference(ws, min_col=2, min_row=2, max_row=2 + n_items)
        # Chart Total and first two cut cols
        chart = BarChart()
        chart.type = "bar"
        chart.grouping = "clustered"
        chart.title = "Importance Scores: Total vs Subgroups"
        chart.y_axis.title = "Item"
        chart.x_axis.title = "Importance (0–100)"

        for data_col in range(3, min(6, len(cols) + 1)):
            data_ref = Reference(ws, min_col=data_col, min_row=2, max_row=2 + n_items)
            chart.add_data(data_ref, titles_from_data=True)
            chart.series[-1].graphicalProperties.solidFill = BAR_COLOURS[(data_col - 3) % len(BAR_COLOURS)]

        chart.set_categories(items_ref)
        chart.width = 22
        chart.height = max(12, n_items * 0.7)
        ws.add_chart(chart, f"{get_column_letter(len(cols) + 2)}3")

    _auto_width(ws)
    ws.column_dimensions["A"].width = 8
    ws.column_dimensions["B"].width = 32


# ---------------------------------------------------------------------------
# Sheet 6: Simulator
# ---------------------------------------------------------------------------

def _write_simulator(ws, importance_matrix, items_df, n_respondents_by_cut):
    """
    Interactive simulator using Excel VLOOKUP formulas and data validation.

    Layout:
      - Row 3-4:  Selector area (dropdown for cut, then sub-group)
      - Row 6:    Column headers
      - Row 7+:   Item rows with VLOOKUP formulas that read the selection
      - Row right: Bar chart linked to VLOOKUP results

    The VLOOKUP formula works as follows:
      =VLOOKUP($D$3, scores_lookup_table, col_offset, FALSE)

    Where scores_lookup_table is a named range built from pre-calculated scores.
    """
    ws.sheet_view.showGridLines = False
    ws.sheet_view.zoomScale = 100

    item_map = dict(zip(items_df["item_id"], items_df["item_label"]))
    n_items = len(items_df)

    # ---- Build cut options list ----
    # cols in importance_matrix: [item_id, item_label, Total, gender_Male, ...]
    cut_cols = [c for c in importance_matrix.columns if c not in ("item_id", "item_label")]
    cut_labels = [c.replace("_", " ") for c in cut_cols]

    # ---- Title ----
    ws.row_dimensions[1].height = 10
    _header_style(ws, 2, 1, 9,
                  "MAXDIFF IMPORTANCE SIMULATOR",
                  bg="sim_header", fg="FFFFFF", size=14)
    ws.row_dimensions[2].height = 32

    _header_style(ws, 3, 1, 9,
                  "Select a subgroup below to see their importance scores. "
                  "The chart and table update automatically via Excel formulas.",
                  bg="simulator_bg", fg="5C3317", size=10, merge=True)
    ws.row_dimensions[3].height = 18

    # ---- Selector ----
    ws.row_dimensions[4].height = 8
    ws.row_dimensions[5].height = 28

    label_c = ws.cell(row=5, column=1, value="SELECT SUBGROUP ▶")
    label_c.font = _font(bold=True, size=11, colour="5C3317")
    label_c.fill = _fill(COLOURS["simulator_bg"])
    label_c.alignment = Alignment(horizontal="right", vertical="center")
    ws.merge_cells(start_row=5, start_column=1, end_row=5, end_column=2)

    selector_cell = ws.cell(row=5, column=3, value=cut_cols[0])
    selector_cell.font = _font(bold=True, size=12, colour="1F3864")
    selector_cell.fill = _fill("DEEAF1")
    selector_cell.alignment = Alignment(horizontal="center", vertical="center")
    selector_cell.border = Border(
        left=Side(style="medium", color="2E75B6"),
        right=Side(style="medium", color="2E75B6"),
        top=Side(style="medium", color="2E75B6"),
        bottom=Side(style="medium", color="2E75B6"),
    )

    # Data validation: dropdown of cut labels
    dv_list = ",".join([f'"{c}"' for c in cut_cols])
    dv = DataValidation(
        type="list",
        formula1=dv_list,
        allow_blank=False,
        showDropDown=False,
    )
    dv.error = "Please select a valid subgroup from the list."
    dv.errorTitle = "Invalid Selection"
    dv.prompt = "Choose a subgroup to display"
    dv.promptTitle = "Subgroup Selector"
    ws.add_data_validation(dv)
    dv.add(selector_cell)

    n_hint_cell = ws.cell(row=5, column=4, value="← Click the cell to select a subgroup")
    n_hint_cell.font = _font(size=9, italic=True, colour="808080")
    n_hint_cell.fill = _fill(COLOURS["simulator_bg"])
    ws.merge_cells(start_row=5, start_column=4, end_row=5, end_column=9)

    ws.row_dimensions[6].height = 8

    # ---- Table headers ----
    col_headers = ["Rank", "Item", "Importance (0–100)", "Comparison: Total",
                   "vs. Total Δ", "Best % (Total)", "Worst % (Total)"]
    for col, h in enumerate(col_headers, 1):
        c = ws.cell(row=7, column=col, value=h)
        c.fill = _fill(COLOURS["header_dark"])
        c.font = _font(bold=True, size=10, colour="FFFFFF")
        c.alignment = Alignment(horizontal="center", wrap_text=True)
        c.border = _border()
    ws.row_dimensions[7].height = 24

    # ---- Build the lookup table on a hidden sheet ----
    # We'll put the scores_lookup table starting at column K (col 11)
    # Format: [cut_label, item1_score, item2_score, ...]
    # The VLOOKUP formula in the Importance column will look this up

    lookup_start_col = 12  # Column L onwards — scores lookup table
    lookup_start_row = 7

    # Header: cut label
    ws.cell(row=lookup_start_row, column=lookup_start_col, value="LOOKUP TABLE (auto)").font = _font(size=8, italic=True, colour="AAAAAA")
    ws.merge_cells(start_row=lookup_start_row, start_column=lookup_start_col,
                   end_row=lookup_start_row, end_column=lookup_start_col + len(items_df))

    # Row headers: cut names
    for row_offset, cut_col in enumerate(cut_cols):
        r = lookup_start_row + 1 + row_offset
        ws.cell(row=r, column=lookup_start_col, value=cut_col).font = _font(size=8, colour="AAAAAA")
        # Score for each item
        for col_offset, item_row in enumerate(importance_matrix.itertuples(), 1):
            score = getattr(item_row, cut_col, None)
            if score is not None:
                ws.cell(row=r, column=lookup_start_col + col_offset, value=round(float(score), 1))

    # Make the lookup area grey/small
    for r in range(lookup_start_row, lookup_start_row + 1 + len(cut_cols)):
        ws.row_dimensions[r].height = 12

    # ---- Item rows with VLOOKUP ----
    # Sort items by Total importance
    sorted_items = importance_matrix.sort_values("Total", ascending=False)["item_id"].tolist()

    selector_ref = "$C$5"  # cell containing chosen cut label

    for row_offset, item_id in enumerate(sorted_items):
        r = 8 + row_offset
        item_label = item_map.get(item_id, f"Item {item_id}")
        bg = COLOURS["white"] if row_offset % 2 == 0 else COLOURS["row_alt"]

        # Rank (based on VLOOKUP importance, approximate with static for now)
        ws.cell(row=r, column=1, value=row_offset + 1).fill = _fill(bg)
        ws.cell(row=r, column=1).font = _font(bold=True, size=10)
        ws.cell(row=r, column=1).alignment = Alignment(horizontal="center")
        ws.cell(row=r, column=1).border = _border()

        # Item label
        item_c = ws.cell(row=r, column=2, value=item_label)
        item_c.fill = _fill(bg)
        item_c.font = _font(size=10)
        item_c.alignment = Alignment(horizontal="left")
        item_c.border = _border()

        # Importance — VLOOKUP formula
        # The lookup table: lookup_start_row+1..+len(cut_cols) rows, lookup_start_col onward
        # Col offset = row_offset + 1 (since first col of lookup is cut_name)
        # We MATCH item position: items are in a fixed order in the lookup table
        # Simpler: VLOOKUP on cut_label, full table, item_col_offset
        item_col_offset = row_offset + 2  # 1-indexed col in lookup table
        lut_range = (
            f"{get_column_letter(lookup_start_col)}{lookup_start_row + 1}:"
            f"{get_column_letter(lookup_start_col + len(items_df))}{lookup_start_row + len(cut_cols)}"
        )
        importance_formula = (
            f'=IFERROR(VLOOKUP({selector_ref},{lut_range},{item_col_offset},FALSE),"")'
        )
        imp_c = ws.cell(row=r, column=3, value=importance_formula)
        imp_c.fill = _fill(bg)
        imp_c.font = _font(size=11, bold=True)
        imp_c.alignment = Alignment(horizontal="center")
        imp_c.border = _border()

        # Total comparison score (static)
        total_score = importance_matrix.loc[
            importance_matrix["item_id"] == item_id, "Total"
        ].values
        total_val = round(float(total_score[0]), 1) if len(total_score) > 0 else 0.0

        total_c = ws.cell(row=r, column=4, value=total_val)
        total_c.fill = _fill(bg)
        total_c.font = _font(size=10, colour="808080")
        total_c.alignment = Alignment(horizontal="center")
        total_c.border = _border()

        # Delta formula (selected - Total)
        delta_c = ws.cell(row=r, column=5,
                           value=f"=IF(C{r}=\"\",\"\",C{r}-D{r})")
        delta_c.font = _font(size=10)
        delta_c.alignment = Alignment(horizontal="center")
        delta_c.border = _border()

        # Best % and Worst % (static from Total for reference)
        best_rate = importance_matrix.loc[importance_matrix["item_id"] == item_id]
        ws.cell(row=r, column=6, value="").fill = _fill(bg)
        ws.cell(row=r, column=7, value="").fill = _fill(bg)

        ws.row_dimensions[r].height = 18

    # ---- Bar chart linked to simulator VLOOKUP results ----
    n_data_rows = len(sorted_items)
    items_ref_sim = Reference(ws, min_col=2, min_row=7, max_row=7 + n_data_rows)
    data_ref_sim = Reference(ws, min_col=3, min_row=7, max_row=7 + n_data_rows)
    total_ref_sim = Reference(ws, min_col=4, min_row=7, max_row=7 + n_data_rows)

    sim_chart = BarChart()
    sim_chart.type = "bar"
    sim_chart.grouping = "clustered"
    sim_chart.title = "Importance: Selected Subgroup vs Total"
    sim_chart.y_axis.title = "Item"
    sim_chart.x_axis.title = "Importance (0–100)"
    sim_chart.add_data(data_ref_sim, titles_from_data=True)
    sim_chart.add_data(total_ref_sim, titles_from_data=True)
    sim_chart.set_categories(items_ref_sim)
    if sim_chart.series:
        sim_chart.series[0].graphicalProperties.solidFill = COLOURS["bar_orange"]
    if len(sim_chart.series) > 1:
        sim_chart.series[1].graphicalProperties.solidFill = COLOURS["bar_blue"]
    sim_chart.width = 22
    sim_chart.height = max(12, n_data_rows * 0.7)
    ws.add_chart(sim_chart, "I7")

    # ---- Column widths ----
    ws.column_dimensions["A"].width = 7
    ws.column_dimensions["B"].width = 30
    ws.column_dimensions["C"].width = 20
    ws.column_dimensions["D"].width = 20
    ws.column_dimensions["E"].width = 14
    ws.column_dimensions["F"].width = 16
    ws.column_dimensions["G"].width = 16

    # Hide the lookup columns (make them very narrow but functional)
    for col in range(lookup_start_col, lookup_start_col + len(items_df) + 2):
        ws.column_dimensions[get_column_letter(col)].width = 5


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_workbook(
    output_path: str,
    design_df: pd.DataFrame,
    design_stats: Dict,
    responses_df: pd.DataFrame,
    items_df: pd.DataFrame,
    cut_results: Dict[str, pd.DataFrame],
    importance_matrix: pd.DataFrame,
    cut_variables: List[str],
    n_respondents_by_cut: Dict[str, int],
    method_label: str = "Counts",
) -> None:
    """
    Build the complete MaxDiff Excel workbook.

    Parameters
    ----------
    output_path          : Path to write the .xlsx file
    design_df            : Design DataFrame from design.generate_design()
    design_stats         : Stats dict from design.generate_design()
    responses_df         : Raw survey responses
    items_df             : Items with [item_id, item_label]
    cut_results          : Dict of {cut_label: scores_df} from analysis.score_all_cuts()
    importance_matrix    : Wide matrix from analysis.build_importance_matrix()
    cut_variables        : List of demographic variable names
    n_respondents_by_cut : Dict {cut_label: n}
    method_label         : 'Counts' or 'Logit'
    """
    wb = Workbook()
    wb.remove(wb.active)  # remove default sheet

    # 1. How It Works
    ws_how = wb.create_sheet("HOW_IT_WORKS")
    ws_how.sheet_properties.tabColor = COLOURS["accent_orange"]
    _write_how_it_works(ws_how, design_stats["n_items"],
                        design_stats["k_per_task"],
                        design_stats["n_tasks"],
                        design_stats)

    # 2. Design
    ws_des = wb.create_sheet("Design")
    ws_des.sheet_properties.tabColor = COLOURS["header_mid"]
    _write_design(ws_des, design_df, items_df, design_stats)

    # 3. Raw Data
    ws_raw = wb.create_sheet("Raw_Data")
    ws_raw.sheet_properties.tabColor = COLOURS["mid_grey"]
    _write_raw_data(ws_raw, responses_df, items_df)

    # 4. Importance Total
    total_scores = cut_results.get("Total", pd.DataFrame())
    n_total = n_respondents_by_cut.get("Total", 0)
    ws_tot = wb.create_sheet("Importance_Total")
    ws_tot.sheet_properties.tabColor = COLOURS["accent_green"]
    if not total_scores.empty:
        _write_importance_total(ws_tot, total_scores, items_df, n_total, method_label)

    # 5. Importance by Cuts
    ws_cuts = wb.create_sheet("Importance_Cuts")
    ws_cuts.sheet_properties.tabColor = COLOURS["bar_orange"]
    _write_importance_cuts(ws_cuts, importance_matrix, items_df,
                           cut_variables, n_respondents_by_cut)

    # 6. Simulator
    ws_sim = wb.create_sheet("Simulator")
    ws_sim.sheet_properties.tabColor = COLOURS["accent_orange"]
    _write_simulator(ws_sim, importance_matrix, items_df, n_respondents_by_cut)

    wb.save(output_path)
    print(f"✓  Workbook saved: {output_path}")
