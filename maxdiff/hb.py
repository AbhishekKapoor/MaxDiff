"""
Hierarchical Bayes (HB) MaxDiff Estimation
============================================
Replicates Sawtooth Software's HB approach using MCMC
(Gibbs sampling for population parameters + Metropolis-Hastings
for individual utilities).

MODEL
------
  Individual utilities:    beta_i  ~ MVN(mu, Sigma)
  Population mean:         mu      ~ MVN(0, 100·I)           [vague prior]
  Population covariance:   Sigma   ~ IW(n_items+3, 2·I)     [unit-scale prior]
  MaxDiff likelihood:      best & worst choices as multinomial logit

WHY HB NEEDS MANY DESIGN VERSIONS
------------------------------------
Aggregate Logit pools all respondents → 2–10 versions suffice.
HB estimates *individual* utilities, requiring each respondent's
data to pull their posterior away from the population mean.
Unique card arrangements (many versions) help the model distinguish
individual preferences from population noise.
Recommended: n_versions ≈ n_respondents (as Sawtooth does by default).

MCMC SCHEDULE (per iteration)
-------------------------------
1. Update each beta_i: random-walk MH with adaptive step size
2. Update mu:   Gibbs draw from conjugate Normal posterior
3. Update Sigma: Gibbs draw from conjugate Inverse-Wishart posterior

POST-PROCESSING
----------------
- Mean-centre each respondent's utilities (MaxDiff only identifies
  differences, not the level)
- Rescale to 0–100 using population-level min/max (comparable across
  respondents)
"""

import numpy as np
from scipy.stats import invwishart
from typing import Callable, Dict, List, Optional, Tuple
import pandas as pd


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def hb_maxdiff(
    responses_df: pd.DataFrame,
    design_df: pd.DataFrame,
    n_items: int,
    n_iter: int = 10_000,
    burn_in: int = 2_000,
    thin: int = 5,
    init_step: float = 0.5,
    seed: int = 42,
    progress_fn: Optional[Callable[[int, int], None]] = None,
) -> Dict:
    """
    Run HB MaxDiff MCMC and return individual-level utilities.

    Parameters
    ----------
    responses_df : columns [respondent_id, version, task, best_item, worst_item]
    design_df    : columns [version, task, position_1 …]
    n_items      : total attributes
    n_iter       : total MCMC draws (including burn-in)
    burn_in      : draws to discard at start
    thin         : keep every Nth post-burn-in draw (reduces autocorrelation)
    init_step    : initial MH proposal SD (adapted automatically)
    seed         : RNG seed
    progress_fn  : optional callback(current_iter, total_iter)

    Returns
    -------
    dict with keys:
      individual_df    – DataFrame(respondent_id, <item_label>…) scaled 0-100
      raw_utilities    – ndarray (n_resp, n_items), mean-centred
      pop_mean         – ndarray (n_items,) – population mean utility
      pop_cov          – ndarray (n_items, n_items) – population covariance
      acceptance_rates – list[float] per respondent
      mean_accept_rate – float (ideal: 0.25–0.45)
      respondent_ids   – list of IDs in array row order
      n_draws          – number of stored post-burn-in draws
    """
    rng = np.random.default_rng(seed)

    # ── Build design lookup (version, task) → [item_idx, ...] 0-indexed ──
    pos_cols = sorted(c for c in design_df.columns if c.startswith("position_"))
    design_lookup: Dict[Tuple[int, int], List[int]] = {}
    for _, row in design_df.iterrows():
        ver  = int(row.get("version", 1))
        task = int(row["task"])
        design_lookup[(ver, task)] = [int(row[pc]) - 1 for pc in pos_cols]

    # ── Respondent index ──
    respondent_ids = sorted(responses_df["respondent_id"].unique())
    n_resp   = len(respondent_ids)
    resp_idx = {rid: i for i, rid in enumerate(respondent_ids)}

    # ── Per-respondent task list: (best_0idx, worst_0idx, set_0idx_list) ──
    resp_tasks: List[List[Tuple[int, int, List[int]]]] = [[] for _ in range(n_resp)]
    for _, row in responses_df.iterrows():
        ri   = resp_idx[row["respondent_id"]]
        ver  = int(row.get("version", 1))
        task = int(row["task"])
        bi   = int(row["best_item"])  - 1
        wi   = int(row["worst_item"]) - 1
        key  = (ver, task)
        if key in design_lookup and 0 <= bi < n_items and 0 <= wi < n_items:
            resp_tasks[ri].append((bi, wi, design_lookup[key]))

    # ── Prior hyper-parameters ──
    V0_inv = np.eye(n_items) * 0.01      # precision of mu prior (var = 100 per item)
    nu0    = float(n_items + 3)           # IW df (weakly informative)
    S0     = 2.0 * np.eye(n_items)       # IW scale → E[Sigma] ≈ I

    # ── Initialise chain ──
    betas     = rng.standard_normal((n_resp, n_items)) * 0.1
    mu        = np.zeros(n_items)
    Sigma     = np.eye(n_items)
    Sigma_inv = np.eye(n_items)

    # Adaptive MH (per respondent)
    steps         = np.full(n_resp, init_step)
    accept_counts = np.zeros(n_resp, dtype=float)
    _ADAPT_EVERY  = 200    # iterations between step-size adjustments

    # Storage buffers
    n_store   = max(1, (n_iter - burn_in) // thin)
    beta_store = np.zeros((n_store, n_resp, n_items))
    mu_store   = np.zeros((n_store, n_items))
    store_ptr  = 0

    # ── MCMC ──
    for it in range(n_iter):

        # ── Step 1: Update individual betas (MH) ──
        for i in range(n_resp):
            proposal = betas[i] + rng.standard_normal(n_items) * steps[i]

            ll_curr = _maxdiff_loglik(betas[i], resp_tasks[i])
            ll_prop = _maxdiff_loglik(proposal,  resp_tasks[i])

            diff_c  = betas[i] - mu
            diff_p  = proposal  - mu
            lp_curr = -0.5 * (diff_c @ Sigma_inv @ diff_c)
            lp_prop = -0.5 * (diff_p @ Sigma_inv @ diff_p)

            log_alpha = (ll_prop + lp_prop) - (ll_curr + lp_curr)
            if np.log(max(rng.random(), 1e-300)) < log_alpha:
                betas[i] = proposal
                accept_counts[i] += 1.0

        # Adapt step sizes during burn-in
        if 0 < it < burn_in and it % _ADAPT_EVERY == 0:
            rates = accept_counts / it
            steps = np.where(rates < 0.20, steps * 0.75,
                    np.where(rates > 0.50, steps * 1.30, steps))
            steps = np.clip(steps, 0.05, 5.0)

        # ── Step 2: Update mu (conjugate Gibbs) ──
        beta_bar  = betas.mean(axis=0)
        V_n_inv   = V0_inv + n_resp * Sigma_inv
        V_n       = np.linalg.inv(V_n_inv)
        m_n       = V_n @ (n_resp * Sigma_inv @ beta_bar)
        try:
            L     = np.linalg.cholesky(_symmetrise(V_n))
            mu    = m_n + L @ rng.standard_normal(n_items)
        except np.linalg.LinAlgError:
            mu    = m_n.copy()

        # ── Step 3: Update Sigma (conjugate IW Gibbs) ──
        dev    = betas - mu[np.newaxis, :]
        S_n    = S0 + dev.T @ dev
        nu_n   = nu0 + n_resp
        try:
            Sigma     = invwishart.rvs(
                df=nu_n, scale=_symmetrise(S_n),
                random_state=int(rng.integers(0, 2**31)),
            )
            Sigma_inv = np.linalg.inv(Sigma)
        except Exception:
            pass   # keep previous Sigma if numerical issues

        # ── Store post-burn-in draws ──
        if it >= burn_in and (it - burn_in) % thin == 0 and store_ptr < n_store:
            beta_store[store_ptr] = betas
            mu_store[store_ptr]   = mu
            store_ptr += 1

        if progress_fn is not None and it % 500 == 0:
            progress_fn(it, n_iter)

    # ── Posterior means ──
    n_actual       = store_ptr
    raw_utilities  = beta_store[:n_actual].mean(axis=0)  # (n_resp, n_items)
    pop_mean_raw   = mu_store[:n_actual].mean(axis=0)

    # Mean-centre per respondent (removes arbitrary level)
    raw_utilities -= raw_utilities.mean(axis=1, keepdims=True)

    # Rescale to 0-100 using population-level range (consistent scale)
    pop_min = raw_utilities.min()
    pop_max = raw_utilities.max()
    pop_rng = pop_max - pop_min
    scaled  = (raw_utilities - pop_min) / pop_rng * 100 if pop_rng > 0 else np.full_like(raw_utilities, 50.0)

    # Build individual DataFrame with item labels
    ind_df = pd.DataFrame({"respondent_id": respondent_ids})
    id_to_label = dict(zip(range(n_items), [f"item_{i+1}" for i in range(n_items)]))
    for j in range(n_items):
        ind_df[f"item_{j+1}"] = np.round(scaled[:, j], 2)

    accept_rates = (accept_counts / n_iter).tolist()

    return {
        "individual_df":    ind_df,
        "raw_utilities":    raw_utilities,
        "pop_mean":         pop_mean_raw,
        "pop_cov":          Sigma,
        "acceptance_rates": accept_rates,
        "mean_accept_rate": float(np.mean(accept_rates)),
        "respondent_ids":   list(respondent_ids),
        "n_draws":          n_actual,
        "n_items":          n_items,
    }


def hb_aggregate_scores(
    raw_utilities: np.ndarray,
    items_df: pd.DataFrame,
    mask: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Aggregate HB individual utilities to group-level importance scores (0-100).

    Parameters
    ----------
    raw_utilities : (n_resp, n_items) mean-centred posterior means
    items_df      : DataFrame with item_id, item_label
    mask          : optional boolean array to select a respondent subset (for cuts)
    """
    utils = raw_utilities[mask] if mask is not None else raw_utilities
    if len(utils) == 0:
        return pd.DataFrame()

    mean_u = utils.mean(axis=0)
    mn, mx = mean_u.min(), mean_u.max()
    rng_   = mx - mn
    imp    = (mean_u - mn) / rng_ * 100 if rng_ > 0 else np.full(len(mean_u), 50.0)

    rows = [
        {
            "item_id":          int(items_df.iloc[j]["item_id"]),
            "item_label":       items_df.iloc[j]["item_label"],
            "importance_0_100": round(float(imp[j]), 1),
        }
        for j in range(len(items_df))
    ]
    df = pd.DataFrame(rows)
    df["rank"] = df["importance_0_100"].rank(ascending=False, method="min").astype(int)
    return df.sort_values("importance_0_100", ascending=False).reset_index(drop=True)


def hb_cut_scores(
    raw_utilities: np.ndarray,
    responses_df: pd.DataFrame,
    items_df: pd.DataFrame,
    respondent_ids: list,
    cut_variables: List[str],
) -> Dict[str, pd.DataFrame]:
    """
    Compute group-level importance scores from HB utilities for all cuts.
    """
    resp_row = {rid: i for i, rid in enumerate(respondent_ids)}

    # Map each respondent to its cut values
    resp_cuts = (
        responses_df[["respondent_id"] + [c for c in cut_variables if c in responses_df.columns]]
        .drop_duplicates("respondent_id")
        .set_index("respondent_id")
    )

    results: Dict[str, pd.DataFrame] = {}
    results["Total"] = hb_aggregate_scores(raw_utilities, items_df)

    for var in cut_variables:
        if var not in resp_cuts.columns:
            continue
        for val in sorted(resp_cuts[var].dropna().unique()):
            rids_in_cut = resp_cuts.index[resp_cuts[var] == val].tolist()
            idxs = np.array([resp_row[r] for r in rids_in_cut if r in resp_row])
            if len(idxs) < 5:
                continue
            mask = np.zeros(len(respondent_ids), dtype=bool)
            mask[idxs] = True
            label = f"{var}_{val}"
            df    = hb_aggregate_scores(raw_utilities, items_df, mask=mask)
            if not df.empty:
                results[label] = df

    return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _maxdiff_loglik(beta: np.ndarray, tasks: List[Tuple]) -> float:
    """Log-likelihood for one respondent's MaxDiff responses (log-sum-exp stable)."""
    if not tasks:
        return 0.0
    ll = 0.0
    for best_idx, worst_idx, set_idxs in tasks:
        u = beta[set_idxs]

        # Best choice: log P(best | set)
        u_s = u - u.max()
        log_z = np.log(np.sum(np.exp(u_s))) + u.max()
        ll += beta[best_idx] - log_z

        # Worst choice: equivalent to best on negated utils, excluding chosen best
        remaining = [i for i in set_idxs if i != best_idx]
        if remaining:
            neg   = -beta[remaining]
            neg_s = neg - neg.max()
            log_z_w = np.log(np.sum(np.exp(neg_s))) + neg.max()
            ll += (-beta[worst_idx]) - log_z_w

    return ll


def _symmetrise(M: np.ndarray) -> np.ndarray:
    """Force exact symmetry to avoid numerical drift."""
    return (M + M.T) * 0.5
