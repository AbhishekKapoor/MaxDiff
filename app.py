"""
MaxDiff Web Interface
=====================
Local Flask app providing a two-tab UI:
  Tab 1 — Design: enter attributes + study settings → download design CSV
  Tab 2 — Analysis: upload response data → scores, simulator, downloads

Run with:
    python app.py
Then open http://localhost:5000
"""

import os
import sys
import io
import threading

import numpy as np
import pandas as pd
from flask import (
    Flask, render_template, request, jsonify,
    make_response, send_file,
)

sys.path.insert(0, os.path.dirname(__file__))

from maxdiff.design import generate_design_from_attributes
from maxdiff.analysis import (
    score_counts, score_logit, score_all_cuts,
    build_importance_matrix, compute_summary_stats,
)
from maxdiff.excel_output import build_workbook
from maxdiff.hb import hb_maxdiff, hb_aggregate_scores, hb_cut_scores

app = Flask(__name__)

# ---------------------------------------------------------------------------
# In-memory state (single-user local app)
# ---------------------------------------------------------------------------
_state: dict = {
    "items_df":    None,
    "design_df":   None,
    "stats":       None,
    "analysis":    None,
}

_hb_lock = threading.Lock()
_hb_state: dict = {
    "status":       "idle",   # idle | running | done | error
    "current_iter": 0,
    "total_iter":   0,
    "result":       None,
    "error":        "",
}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/design", methods=["POST"])
def api_design():
    data = request.get_json(force=True)
    study_name  = (data.get("study_name") or "MaxDiff Study").strip()
    n_resp      = max(10, int(data.get("n_respondents", 300)))
    attrs_raw   = data.get("attributes", "")
    attributes  = [
        a.strip() for a in attrs_raw.split("\n")
        if a.strip() and not a.strip().startswith("#")
    ]
    if len(attributes) < 4:
        return jsonify({"success": False, "error": "Please enter at least 4 attributes."}), 400

    k = data.get("k_per_task") or None
    t = data.get("n_tasks")    or None
    v = data.get("n_versions") or None
    k = int(k) if k else None
    t = int(t) if t else None
    v = max(1, int(v)) if v else None

    # Sawtooth research shows HB accuracy plateaus after ~20-30 versions;
    # 50 is sufficient for any method. Cap auto-default at 50 (never n_respondents).
    effective_versions = v if v is not None else min(50, n_resp)

    try:
        items_df, design_df, stats = generate_design_from_attributes(
            attributes=attributes,
            n_respondents=n_resp,
            k_per_task=k,
            n_tasks=t,
            seed=42,
            n_versions=effective_versions,
        )
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500

    # Warn when user-specified tasks give < 3 appearances per item
    n_items_actual  = int(stats["n_items"])
    k_actual        = int(stats["k_per_task"])
    t_actual        = int(stats["n_tasks"])
    appearances     = (t_actual * k_actual) / n_items_actual
    design_warnings = []
    if appearances < 2:
        min_tasks = int(np.ceil(2 * n_items_actual / k_actual))
        design_warnings.append(
            f"Only {appearances:.1f} appearances per item — too few for reliable estimates. "
            f"Increase tasks to at least {min_tasks}."
        )
    elif appearances < 3:
        min_tasks = int(np.ceil(3 * n_items_actual / k_actual))
        design_warnings.append(
            f"{appearances:.1f} appearances per item — below Sawtooth's recommended minimum of 3. "
            f"Increase tasks to {min_tasks} for reliable HB estimates."
        )

    _state["items_df"]  = items_df.copy()
    _state["design_df"] = design_df.copy()
    _state["stats"]     = stats

    design_csv = _build_survey_csv(design_df, items_df)
    items_csv  = items_df.to_csv(index=False)

    return jsonify({
        "success":       True,
        "study_name":    study_name,
        "design_csv":    design_csv,
        "items_csv":     items_csv,
        "n_items":       int(stats["n_items"]),
        "k_per_task":    int(stats["k_per_task"]),
        "n_tasks":       int(stats["n_tasks"]),
        "n_versions":    int(stats["n_versions"]),
        "design_type":   str(stats["design_type"]),
        "is_orthogonal": bool(stats["is_orthogonal"]),
        "pair_cv":       round(float(stats["pair_cv"]), 4),
        "mean_appearances": round(float(stats["mean_appearances"]), 1),
        "attributes":    attributes,
        "design_warnings": design_warnings,
    })


@app.route("/api/analysis", methods=["POST"])
def api_analysis():
    responses_file = request.files.get("responses")
    items_file     = request.files.get("items")
    design_file    = request.files.get("design")
    method         = request.form.get("method", "counts")
    study_name     = request.form.get("study_name", "MaxDiff Study")

    if not responses_file:
        return jsonify({"success": False, "error": "responses.csv is required."}), 400
    if not items_file and _state["items_df"] is None:
        return jsonify({"success": False,
                        "error": "items.csv is required (or run the Design tab first)."}), 400

    try:
        responses_df = pd.read_csv(responses_file)
    except Exception as exc:
        return jsonify({"success": False, "error": f"Cannot read responses: {exc}"}), 400

    if items_file:
        try:
            items_df = pd.read_csv(items_file)
        except Exception as exc:
            return jsonify({"success": False, "error": f"Cannot read items: {exc}"}), 400
    else:
        items_df = _state["items_df"].copy()

    required_cols = ["respondent_id", "version", "task", "best_item", "worst_item"]
    missing = [c for c in required_cols if c not in responses_df.columns]
    if missing:
        return jsonify({"success": False,
                        "error": f"responses.csv is missing columns: {missing}"}), 400
    if "item_id" not in items_df.columns or "item_label" not in items_df.columns:
        return jsonify({"success": False,
                        "error": "items.csv must have columns: item_id, item_label"}), 400

    n_items       = len(items_df)
    label_to_id   = dict(zip(items_df["item_label"], items_df["item_id"]))
    id_to_label   = dict(zip(items_df["item_id"],    items_df["item_label"]))

    # Convert label → id if responses used labels
    for col in ["best_item", "worst_item"]:
        if responses_df[col].dtype == object:
            responses_df[col] = responses_df[col].map(label_to_id)
    responses_df["best_item"]  = pd.to_numeric(responses_df["best_item"],  errors="coerce").fillna(0).astype(int)
    responses_df["worst_item"] = pd.to_numeric(responses_df["worst_item"], errors="coerce").fillna(0).astype(int)

    # Load / generate design
    if design_file:
        try:
            raw = pd.read_csv(design_file)
            item_cols = [c for c in raw.columns if c.startswith("item_") and c != "item_id"]
            design_df = raw.copy()
            for col in item_cols:
                pos_col = "position_" + col.split("_")[1]
                design_df[pos_col] = design_df[col].map(label_to_id)
        except Exception as exc:
            return jsonify({"success": False, "error": f"Cannot read design: {exc}"}), 400
    elif _state["design_df"] is not None:
        design_df = _state["design_df"].copy()
    else:
        attrs  = items_df["item_label"].tolist()
        n_seen = max(responses_df["respondent_id"].nunique(), 100)
        _, design_df, _ = generate_design_from_attributes(attrs, n_seen, seed=42, n_versions=2)

    # Identify demographic cut variables
    non_cut = set(required_cols)
    cut_vars = [
        c for c in responses_df.columns
        if c not in non_cut and responses_df[c].nunique() <= 20
    ]

    score_fn = score_counts if method == "counts" else score_logit

    total_scores = score_fn(responses_df, design_df, n_items)
    if total_scores.empty:
        return jsonify({"success": False,
                        "error": "Scoring returned no results — check that item IDs match."}), 400
    total_scores["item_label"] = total_scores["item_id"].map(id_to_label)

    cut_results       = score_all_cuts(responses_df, design_df, n_items, cut_vars, method=method)
    importance_matrix = build_importance_matrix(cut_results, items_df)
    summary_stats     = compute_summary_stats(responses_df, cut_vars)
    n_by_cut          = dict(zip(summary_stats["cut"], summary_stats["n"]))
    ind_scores_df     = _score_individual_respondents(responses_df, design_df, n_items, items_df)

    _state["analysis"] = {
        "study_name":       study_name,
        "responses_df":     responses_df,
        "items_df":         items_df,
        "design_df":        design_df,
        "stats":            _state.get("stats") or {},
        "total_scores":     total_scores,
        "cut_results":      cut_results,
        "importance_matrix": importance_matrix,
        "ind_scores_df":    ind_scores_df,
        "cut_vars":         cut_vars,
        "n_by_cut":         n_by_cut,
        "method":           method,
    }

    # --- Build JSON response ---
    scores_out = [
        {
            "item_id":    int(r["item_id"]),
            "label":      r["item_label"],
            "importance": float(r["importance_0_100"]),
            "rank":       int(r["rank"]),
        }
        for _, r in total_scores.iterrows()
    ]

    # cuts_data: {item_label: {cut_label: importance, ...}}
    cut_cols = [c for c in importance_matrix.columns if c not in ("item_id", "item_label")]
    cuts_data = {}
    for _, row in importance_matrix.iterrows():
        lbl = row["item_label"]
        cuts_data[lbl] = {
            col: (round(float(row[col]), 1) if pd.notna(row.get(col)) else None)
            for col in cut_cols
        }

    return jsonify({
        "success":       True,
        "scores":        scores_out,
        "cuts":          cut_cols,
        "cuts_data":     cuts_data,
        "n_respondents": int(responses_df["respondent_id"].nunique()),
        "n_items":       n_items,
        "method":        method,
        "ind_csv":       ind_scores_df.to_csv(index=False),
        "n_by_cut":      {k: int(v) for k, v in n_by_cut.items()},
    })


# ---------------------------------------------------------------------------
# Download endpoints
# ---------------------------------------------------------------------------

@app.route("/download/design")
def download_design():
    if _state["design_df"] is None:
        return "No design generated yet.", 404
    csv_data = _build_survey_csv(_state["design_df"], _state["items_df"])
    resp = make_response(csv_data)
    resp.headers["Content-Disposition"] = "attachment; filename=design_output.csv"
    resp.headers["Content-Type"] = "text/csv"
    return resp


@app.route("/download/items")
def download_items():
    if _state["items_df"] is None:
        return "No design generated yet.", 404
    resp = make_response(_state["items_df"].to_csv(index=False))
    resp.headers["Content-Disposition"] = "attachment; filename=items.csv"
    resp.headers["Content-Type"] = "text/csv"
    return resp


@app.route("/download/importance")
def download_importance():
    if _state["analysis"] is None:
        return "No analysis run yet.", 404
    resp = make_response(_state["analysis"]["ind_scores_df"].to_csv(index=False))
    resp.headers["Content-Disposition"] = "attachment; filename=importance_by_respondent.csv"
    resp.headers["Content-Type"] = "text/csv"
    return resp


@app.route("/download/excel")
def download_excel():
    if _state["analysis"] is None:
        return "No analysis run yet.", 404
    a = _state["analysis"]
    os.makedirs("output", exist_ok=True)
    out_path = os.path.join("output", "MaxDiff_Results.xlsx")
    method_label = "Aggregate Logit" if a["method"] == "logit" else "Counts"
    build_workbook(
        output_path=out_path,
        design_df=a["design_df"],
        design_stats=a["stats"] if a["stats"] else {
            "design_type": "Balanced", "n_items": len(a["items_df"]),
            "k_per_task": 5, "n_tasks": 12, "n_versions": 2,
            "is_orthogonal": False, "pair_cv": 0.1,
            "mean_appearances": 3.0, "min_appearances": 3, "max_appearances": 4,
            "item_balance_ratio": 0.75, "pair_counts_min": 1,
            "pair_counts_max": 2, "pair_counts_mean": 1.5,
        },
        responses_df=a["responses_df"],
        items_df=a["items_df"],
        cut_results=a["cut_results"],
        importance_matrix=a["importance_matrix"],
        cut_variables=a["cut_vars"],
        n_respondents_by_cut=a["n_by_cut"],
        method_label=method_label,
    )
    return send_file(
        os.path.abspath(out_path),
        as_attachment=True,
        download_name="MaxDiff_Results.xlsx",
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_survey_csv(design_df: pd.DataFrame, items_df: pd.DataFrame) -> str:
    """Return a user-friendly design CSV with item labels instead of IDs."""
    item_map  = dict(zip(items_df["item_id"], items_df["item_label"]))
    pos_cols  = sorted([c for c in design_df.columns if c.startswith("position_")])
    survey_df = design_df[["version", "task"]].copy()
    for i, col in enumerate(pos_cols, start=1):
        survey_df[f"item_{i}"] = design_df[col].map(item_map)
    return survey_df.to_csv(index=False)


def _score_individual_respondents(
    responses_df: pd.DataFrame,
    design_df: pd.DataFrame,
    n_items: int,
    items_df: pd.DataFrame,
) -> pd.DataFrame:
    """Individual counting scores (0–100) for every respondent × attribute."""
    pos_cols = sorted([c for c in design_df.columns if c.startswith("position_")])
    design_lookup: dict = {}
    for _, row in design_df.iterrows():
        ver  = int(row.get("version", 1))
        task = int(row["task"])
        design_lookup[(ver, task)] = [int(row[pc]) for pc in pos_cols]

    id_to_label = dict(zip(items_df["item_id"], items_df["item_label"]))
    rows = []

    for resp_id in sorted(responses_df["respondent_id"].unique()):
        rdata = responses_df[responses_df["respondent_id"] == resp_id]
        best_c = np.zeros(n_items + 1)
        worst_c = np.zeros(n_items + 1)
        apps   = np.zeros(n_items + 1)

        for _, rrow in rdata.iterrows():
            ver  = int(rrow.get("version", 1))
            task = int(rrow["task"])
            bi   = int(rrow["best_item"])
            wi   = int(rrow["worst_item"])
            if (ver, task) in design_lookup:
                for item in design_lookup[(ver, task)]:
                    apps[item] += 1
                best_c[bi]  += 1
                worst_c[wi] += 1

        raw = {i: ((best_c[i] - worst_c[i]) / apps[i] if apps[i] > 0 else 0.0)
               for i in range(1, n_items + 1)}
        vals = list(raw.values())
        mn, mx = min(vals), max(vals)
        rng = mx - mn

        row_data: dict = {"respondent_id": resp_id}
        for i in range(1, n_items + 1):
            lbl = id_to_label.get(i, f"Item {i}")
            row_data[lbl] = round((raw[i] - mn) / rng * 100, 1) if rng > 0 else 50.0
        rows.append(row_data)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# HB routes
# ---------------------------------------------------------------------------

def _run_hb_thread(responses_df, design_df, n_items, items_df, cut_vars,
                   n_iter, burn_in, thin, study_name):
    """Background thread: runs HB MCMC and stores result in _hb_state."""
    id_to_label = dict(zip(items_df["item_id"], items_df["item_label"]))

    def progress_fn(current, total):
        with _hb_lock:
            _hb_state["current_iter"] = current
            _hb_state["total_iter"]   = total

    try:
        hb_result = hb_maxdiff(
            responses_df=responses_df,
            design_df=design_df,
            n_items=n_items,
            n_iter=n_iter,
            burn_in=burn_in,
            thin=thin,
            progress_fn=progress_fn,
        )

        raw_utils      = hb_result["raw_utilities"]
        respondent_ids = hb_result["respondent_ids"]

        total_scores_df = hb_aggregate_scores(raw_utils, items_df)
        total_scores_df["item_label"] = total_scores_df["item_id"].map(id_to_label)

        cut_scores_dict = hb_cut_scores(raw_utils, responses_df, items_df,
                                        respondent_ids, cut_vars)

        # Build importance_matrix and n_by_cut
        all_results      = {"Total": total_scores_df, **cut_scores_dict}
        importance_matrix = build_importance_matrix(all_results, items_df)
        summary_stats     = compute_summary_stats(responses_df, cut_vars)
        n_by_cut          = dict(zip(summary_stats["cut"], summary_stats["n"]))

        # Build individual DataFrame with actual item labels
        ind_df = hb_result["individual_df"].copy()
        rename = {f"item_{j+1}": id_to_label.get(j+1, f"item_{j+1}")
                  for j in range(n_items)}
        ind_df.rename(columns=rename, inplace=True)

        with _hb_lock:
            _hb_state["result"] = {
                "study_name":       study_name,
                "hb_result":        hb_result,
                "total_scores":     total_scores_df,
                "cut_scores":       cut_scores_dict,
                "importance_matrix": importance_matrix,
                "items_df":         items_df,
                "cut_vars":         cut_vars,
                "n_by_cut":         n_by_cut,
                "ind_df":           ind_df,
                "n_items":          n_items,
                "n_respondents":    responses_df["respondent_id"].nunique(),
            }
            _hb_state["status"]       = "done"
            _hb_state["current_iter"] = n_iter
            _hb_state["total_iter"]   = n_iter

    except Exception as exc:
        with _hb_lock:
            _hb_state["status"] = "error"
            _hb_state["error"]  = str(exc)


@app.route("/api/hb_start", methods=["POST"])
def api_hb_start():
    with _hb_lock:
        if _hb_state["status"] == "running":
            return jsonify({"success": False, "error": "HB is already running."}), 400

    responses_file = request.files.get("responses")
    items_file     = request.files.get("items")
    design_file    = request.files.get("design")
    study_name     = request.form.get("study_name", "MaxDiff Study")
    n_iter  = int(request.form.get("n_iter",  10000))
    burn_in = int(request.form.get("burn_in", 2000))
    thin    = int(request.form.get("thin",    5))

    if not responses_file:
        return jsonify({"success": False, "error": "responses.csv is required."}), 400
    if not items_file and _state["items_df"] is None:
        return jsonify({"success": False,
                        "error": "items.csv is required (or run the Design tab first)."}), 400

    try:
        responses_df = pd.read_csv(responses_file)
    except Exception as exc:
        return jsonify({"success": False, "error": f"Cannot read responses: {exc}"}), 400

    if items_file:
        try:
            items_df = pd.read_csv(items_file)
        except Exception as exc:
            return jsonify({"success": False, "error": f"Cannot read items: {exc}"}), 400
    else:
        items_df = _state["items_df"].copy()

    required_cols = ["respondent_id", "version", "task", "best_item", "worst_item"]
    missing = [c for c in required_cols if c not in responses_df.columns]
    if missing:
        return jsonify({"success": False,
                        "error": f"responses.csv is missing columns: {missing}"}), 400
    if "item_id" not in items_df.columns or "item_label" not in items_df.columns:
        return jsonify({"success": False,
                        "error": "items.csv must have columns: item_id, item_label"}), 400

    n_items     = len(items_df)
    label_to_id = dict(zip(items_df["item_label"], items_df["item_id"]))

    for col in ["best_item", "worst_item"]:
        if responses_df[col].dtype == object:
            responses_df[col] = responses_df[col].map(label_to_id)
    responses_df["best_item"]  = pd.to_numeric(responses_df["best_item"],  errors="coerce").fillna(0).astype(int)
    responses_df["worst_item"] = pd.to_numeric(responses_df["worst_item"], errors="coerce").fillna(0).astype(int)

    if design_file:
        try:
            raw = pd.read_csv(design_file)
            item_cols = [c for c in raw.columns if c.startswith("item_") and c != "item_id"]
            design_df = raw.copy()
            for col in item_cols:
                pos_col = "position_" + col.split("_")[1]
                design_df[pos_col] = design_df[col].map(label_to_id)
        except Exception as exc:
            return jsonify({"success": False, "error": f"Cannot read design: {exc}"}), 400
    elif _state["design_df"] is not None:
        design_df = _state["design_df"].copy()
    else:
        attrs  = items_df["item_label"].tolist()
        n_seen = max(responses_df["respondent_id"].nunique(), 100)
        _, design_df, _ = generate_design_from_attributes(attrs, n_seen, seed=42, n_versions=2)

    non_cut = set(required_cols)
    cut_vars = [
        c for c in responses_df.columns
        if c not in non_cut and responses_df[c].nunique() <= 20
    ]

    with _hb_lock:
        _hb_state["status"]       = "running"
        _hb_state["current_iter"] = 0
        _hb_state["total_iter"]   = n_iter
        _hb_state["result"]       = None
        _hb_state["error"]        = ""

    t = threading.Thread(
        target=_run_hb_thread,
        args=(responses_df, design_df, n_items, items_df, cut_vars,
              n_iter, burn_in, thin, study_name),
        daemon=True,
    )
    t.start()

    return jsonify({"success": True, "n_iter": n_iter})


@app.route("/api/hb_progress")
def api_hb_progress():
    with _hb_lock:
        s = dict(_hb_state)
    pct = 0
    if s["total_iter"] > 0:
        pct = round(s["current_iter"] / s["total_iter"] * 100, 1)
    return jsonify({
        "status":       s["status"],
        "current_iter": s["current_iter"],
        "total_iter":   s["total_iter"],
        "pct":          pct,
        "error":        s["error"],
    })


@app.route("/api/hb_result")
def api_hb_result():
    with _hb_lock:
        s = dict(_hb_state)
    if s["status"] != "done":
        return jsonify({"success": False, "error": "HB not complete yet."}), 400

    r         = s["result"]
    hb        = r["hb_result"]
    tot       = r["total_scores"]
    imp_mat   = r["importance_matrix"]
    items_df  = r["items_df"]
    id_to_label = dict(zip(items_df["item_id"], items_df["item_label"]))

    scores_out = [
        {
            "item_id":    int(row["item_id"]),
            "label":      row["item_label"],
            "importance": float(row["importance_0_100"]),
            "rank":       int(row["rank"]),
        }
        for _, row in tot.iterrows()
    ]

    cut_cols  = [c for c in imp_mat.columns if c not in ("item_id", "item_label")]
    cuts_data = {}
    for _, row in imp_mat.iterrows():
        lbl = row["item_label"]
        cuts_data[lbl] = {
            col: (round(float(row[col]), 1) if pd.notna(row.get(col)) else None)
            for col in cut_cols
        }

    return jsonify({
        "success":          True,
        "scores":           scores_out,
        "cuts":             cut_cols,
        "cuts_data":        cuts_data,
        "n_respondents":    int(r["n_respondents"]),
        "n_items":          int(r["n_items"]),
        "method":           "hb",
        "n_by_cut":         {k: int(v) for k, v in r["n_by_cut"].items()},
        "mean_accept_rate": round(float(hb["mean_accept_rate"]) * 100, 1),
        "n_draws":          int(hb["n_draws"]),
    })


@app.route("/download/hb_individual")
def download_hb_individual():
    with _hb_lock:
        s = dict(_hb_state)
    if s["status"] != "done" or s["result"] is None:
        return "No HB analysis complete yet.", 404
    csv_data = s["result"]["ind_df"].to_csv(index=False)
    resp = make_response(csv_data)
    resp.headers["Content-Disposition"] = "attachment; filename=hb_individual_utilities.csv"
    resp.headers["Content-Type"] = "text/csv"
    return resp


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    os.makedirs("templates", exist_ok=True)
    print("\n  MaxDiff Web App")
    print("  ─────────────────────────────────")
    print("  Open http://localhost:5000 in your browser\n")
    app.run(debug=True, port=5000, host="0.0.0.0")
