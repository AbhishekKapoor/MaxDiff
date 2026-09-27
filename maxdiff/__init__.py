"""MaxDiff Analysis Package — replicates Sawtooth Software MaxDiff."""

from .design import (
    generate_design_from_attributes,
    generate_design,
    check_feasibility,
    check_bibd_feasibility,
    find_best_bibd,
    auto_design_params,
    design_to_survey_cards,
    print_design_summary,
)
from .analysis import (
    score_counts, score_logit, score_all_cuts,
    build_importance_matrix, compute_summary_stats,
)
from .excel_output import build_workbook
from .sample_data import generate_sample_items, generate_responses
from .input_parser import parse_study_setup, find_default_setup, load_items_csv

__all__ = [
    "generate_design_from_attributes",
    "generate_design",
    "check_feasibility",
    "check_bibd_feasibility",
    "find_best_bibd",
    "auto_design_params",
    "design_to_survey_cards",
    "print_design_summary",
    "score_counts",
    "score_logit",
    "score_all_cuts",
    "build_importance_matrix",
    "compute_summary_stats",
    "build_workbook",
    "generate_sample_items",
    "generate_responses",
    "parse_study_setup",
    "find_default_setup",
    "load_items_csv",
]
