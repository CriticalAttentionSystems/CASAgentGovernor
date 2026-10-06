#!/usr/bin/env python3
"""
analyze_stochastic_policy_experiment.py

Implements the confirmatory analysis specified across all three
pre-registration documents for RunStochasticPolicyExperiment's output:

  - prereg_cas_governor.md            H1   (governed vs static_good)
  - prereg_addendum_A_disturbance_first.md  H1-B (governed vs disturbance_first)
  - prereg_addendum_B_sham_regulation.md    H1-C (governed vs sham_regulation)

All three are co-primary endpoints, each tested once at nominal alpha=0.05,
at the same pre-registered cell (agent_type=persistent, max_steps=16,
pooled across all 8 tasks). H1 and H1-B use an independent-samples
two-proportion z-test (Fisher's exact substituted if any cell < 5). H1-C
uses McNemar's test instead, NOT the independent-samples test -- see
"IMPORTANT: H1-C is a matched-pairs design" below for why this isn't
optional.

Consumes output/stochastic_policy_runs.csv (as written by the Java
experiment) and produces:

  1. primary_endpoint_result.json / .txt
       All three co-primary endpoints (H1, H1-B, H1-C), plus a
       Cochran-Mantel-Haenszel robustness check on H1 (stratified by
       task_id; NOT part of the confirmatory decision for any endpoint).
       Each endpoint is computed exactly once. Do not re-run this after
       inspecting results with different filters "to see if it's
       significant" -- that defeats the point of pre-registration.

  2. secondary_endpoint_results.csv
       Every other cell for the comparison pairs used above (the other 4
       agent types and 3 other step budgets for disturbance_first and
       sham_regulation; all 5 agent types x 4 step budgets for the
       remaining pairs: governed vs ungoverned, reweight_only vs
       ungoverned, block_only vs ungoverned, governed vs reweight_only,
       governed vs block_only). A `test_type` column marks each row as
       `independent_z` or `mcnemar_paired`. ALL rows share one joint
       Benjamini-Hochberg FDR correction (target FDR = 0.10) -- the sham
       rows are not a separate correction family.

  3. task_breakdown_primary_cell.csv
       Descriptive (non-confirmatory) per-task success rates for the
       primary cell's conditions, including the two adversarial tasks
       (T005, T008) where the discovery-based policy is *expected* to
       underperform -- a sanity check on the omniscience fix, not evidence
       for/against any Hx.

  4. descriptive_grid_summary.csv
       Full descriptive grid (mean success/answered/tool-calls/signals/
       warnings/blocks per agent_type x condition x max_steps),
       independently recomputed in Python as a cross-check against the
       Java-side aggregate CSV.

IMPORTANT -- H1-C is a matched-pairs design, not independent samples:
Every sham_regulation run is CONSTRUCTED from a specific governed run
(same seed, same task -- see RunStochasticPolicyExperiment.runGovernedThenSham
and StochasticToolAgent's sham mode). They are not independent samples, so
applying the independent-samples z-test to them (as an earlier version of
this script did, implicitly, by lumping ("governed","sham_regulation") into
the same generic secondary-comparison machinery as everything else) would
be a real methodological error, not just a stylistic mismatch: it would
misstate the standard error and could produce a materially wrong p-value.
McNemar's test uses only the DISCORDANT pairs (governed right/sham wrong,
or vice versa) and is the correct tool here. See mcnemar_test() and
prereg_addendum_B_sham_regulation.md \u00a73 for the full derivation.

Usage:
    python3 analyze_stochastic_policy_experiment.py \\
        --input output/stochastic_policy_runs.csv \\
        --outdir analysis_output

If --input does not exist, run RunStochasticPolicyExperiment.java first,
or pass --self-test to validate this script against a small synthetic
fixture (does NOT touch the pre-registration; purely a smoke test that the
code runs end-to-end without errors).
"""

import argparse
import csv
import json
import math
import os
import sys
from itertools import product

import numpy as np
from scipy import stats


# ----------------------------------------------------------------------
# Pre-registered constants (must match the three pre-registration
# documents exactly)
# ----------------------------------------------------------------------

PRIMARY_AGENT_TYPE = "persistent"
PRIMARY_MAX_STEPS = 16
PRIMARY_ALPHA = 0.05

# The three co-primary endpoints, each evaluated once at the same cell above.
H1_CONDITIONS = ("governed", "static_good")           # prereg_cas_governor.md
H1B_CONDITIONS = ("governed", "disturbance_first")     # Addendum A
H1C_CONDITIONS = ("governed", "sham_regulation")       # Addendum B (paired/McNemar)

SECONDARY_ALPHA_FDR = 0.10
# Comparison pairs swept across every (agent_type, max_steps) cell for the
# secondary/exploratory analysis. H1B_CONDITIONS and H1C_CONDITIONS are
# included here too (their OTHER 20 cells are exploratory per their
# addenda \u00a74/\u00a76) but the primary cell itself is skipped for those two
# pairs specifically inside secondary_endpoints(), since it already has a
# dedicated primary-endpoint result above and shouldn't be double-counted
# in the exploratory pool.
SECONDARY_COMPARISONS = [
    ("governed", "ungoverned"),
    ("reweight_only", "ungoverned"),
    ("block_only", "ungoverned"),
    ("governed", "reweight_only"),
    ("governed", "block_only"),
    H1B_CONDITIONS,
    H1C_CONDITIONS,
]

ALL_CONDITIONS = [
    "ungoverned", "reweight_only", "block_only", "governed",
    "disturbance_first", "sham_regulation", "static_good",
]
ALL_AGENT_TYPES = ["exploratory", "clinging", "persistent", "impulsive", "noisy"]
ALL_STEP_BUDGETS = [4, 8, 16, 32]


# ----------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------

def load_data(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            r["seed"] = int(r["seed"])
            r["max_steps"] = int(r["max_steps"])
            r["success"] = r["success"].strip().lower() == "true"
            r["answered"] = r["answered"].strip().lower() == "true"
            for numeric_field in (
                "total_tool_calls", "failed_tool_calls", "successful_tool_calls",
                "repeated_failed_actions", "repeated_successful_actions",
                "governor_interventions", "regulatory_signals", "disturbed_signals",
                "recovering_signals", "stuck_signals", "released_signals",
                "steps_after_first_failure", "post_error_tool_burden",
                "soft_warnings", "hard_blocks",
            ):
                if numeric_field in r:
                    r[numeric_field] = int(r[numeric_field])
            rows.append(r)
    if not rows:
        raise ValueError(f"No rows loaded from {path}")
    return rows


def filter_rows(rows, **kwargs):
    out = rows
    for k, v in kwargs.items():
        out = [r for r in out if r[k] == v]
    return out


# ----------------------------------------------------------------------
# Statistics: Wilson CI, two-proportion z-test, Newcombe risk-difference
# CI, log risk-ratio CI, Benjamini-Hochberg FDR, Cochran-Mantel-Haenszel,
# McNemar's test (paired), paired risk-difference CI
# ----------------------------------------------------------------------

def wilson_ci(successes, n, alpha=0.05):
    """Wilson score interval for a single binomial proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    z = stats.norm.ppf(1 - alpha / 2)
    phat = successes / n
    denom = 1 + z ** 2 / n
    center = (phat + z ** 2 / (2 * n)) / denom
    half = (z * math.sqrt((phat * (1 - phat) + z ** 2 / (4 * n)) / n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def two_proportion_ztest(x1, n1, x2, n2):
    """
    Pooled two-proportion z-test for INDEPENDENT samples. Returns
    (z, two_sided_p, used_exact, exact_p_or_None). If either group's
    success or failure count is < 5, also computes Fisher's exact test
    p-value and flags used_exact=True to signal the exact p-value should
    be preferred for the reported p-value, while z/CI machinery is still
    reported for the effect size.

    Do NOT use this for governed vs sham_regulation -- see mcnemar_test().
    """
    p1 = x1 / n1
    p2 = x2 / n2
    p_pool = (x1 + x2) / (n1 + n2)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))

    if se == 0:
        z = 0.0
        p_two_sided = 1.0
    else:
        z = (p1 - p2) / se
        p_two_sided = 2 * (1 - stats.norm.cdf(abs(z)))

    cell_counts = [x1, n1 - x1, x2, n2 - x2]
    used_exact = min(cell_counts) < 5
    exact_p = None
    if used_exact:
        table = [[x1, n1 - x1], [x2, n2 - x2]]
        _, exact_p = stats.fisher_exact(table)

    return z, p_two_sided, used_exact, exact_p


def newcombe_risk_difference_ci(x1, n1, x2, n2, alpha=0.05):
    """
    Newcombe (1998) Method 10 hybrid score CI for the difference of two
    INDEPENDENT proportions. More reliable at small/extreme proportions
    than a simple normal-approximation CI on the risk difference.
    """
    l1, u1 = wilson_ci(x1, n1, alpha)
    l2, u2 = wilson_ci(x2, n2, alpha)
    p1, p2 = x1 / n1, x2 / n2

    diff = p1 - p2
    lo = diff - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = diff + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return diff, lo, hi


def log_risk_ratio_ci(x1, n1, x2, n2, alpha=0.05, correction=0.5):
    """
    Risk ratio (p1/p2) with a log-scale normal-approximation CI, for
    INDEPENDENT samples. Applies a 0.5 continuity correction to all four
    cells if any cell is zero, to avoid division by zero / log(0).
    """
    if x1 == 0 or x2 == 0 or (n1 - x1) == 0 or (n2 - x2) == 0:
        x1c, n1c, x2c, n2c = x1 + correction, n1 + 2 * correction, x2 + correction, n2 + 2 * correction
    else:
        x1c, n1c, x2c, n2c = x1, n1, x2, n2

    p1, p2 = x1c / n1c, x2c / n2c
    rr = p1 / p2
    se_log_rr = math.sqrt((1 / x1c) - (1 / n1c) + (1 / x2c) - (1 / n2c))
    z = stats.norm.ppf(1 - alpha / 2)
    log_rr = math.log(rr)
    lo = math.exp(log_rr - z * se_log_rr)
    hi = math.exp(log_rr + z * se_log_rr)
    return rr, lo, hi


def benjamini_hochberg(pvalues, alpha=0.10):
    """
    Standard BH step-up FDR procedure. Returns (adjusted_pvalues, reject)
    in the ORIGINAL input order.
    """
    m = len(pvalues)
    indexed = sorted(enumerate(pvalues), key=lambda t: t[1])

    adjusted = [0.0] * m
    prev_min = 1.0
    for rank in range(m - 1, -1, -1):
        orig_idx, p = indexed[rank]
        bh_value = p * m / (rank + 1)
        prev_min = min(prev_min, bh_value)
        adjusted[orig_idx] = min(prev_min, 1.0)

    reject = [adjusted[i] <= alpha for i in range(m)]
    return adjusted, reject


def cochran_mantel_haenszel(strata_2x2):
    """
    CMH test for a common odds ratio across strata, with continuity
    correction. strata_2x2: list of (a, b, c, d) tuples, one 2x2 table
    per stratum, where a=group1 success, b=group1 failure, c=group2
    success, d=group2 failure.

    Returns (mh_odds_ratio, chi2_statistic, p_value).
    """
    num_or, den_or = 0.0, 0.0
    sum_a, sum_e, sum_var = 0.0, 0.0, 0.0

    for a, b, c, d in strata_2x2:
        n = a + b + c + d
        if n == 0:
            continue
        num_or += (a * d) / n
        den_or += (b * c) / n

        e_a = (a + b) * (a + c) / n
        var_a = (a + b) * (c + d) * (a + c) * (b + d) / (n ** 2 * (n - 1)) if n > 1 else 0.0

        sum_a += a
        sum_e += e_a
        sum_var += var_a

    mh_or = (num_or / den_or) if den_or != 0 else float("inf")

    if sum_var == 0:
        chi2, p = 0.0, 1.0
    else:
        chi2 = (abs(sum_a - sum_e) - 0.5) ** 2 / sum_var
        p = 1 - stats.chi2.cdf(chi2, df=1)

    return mh_or, chi2, p


def pair_by_seed_task(rows, agent_type, max_steps, condition_a, condition_b):
    """
    Matches rows of condition_a and condition_b within one
    (agent_type, max_steps) cell by (seed, task_id) -- the pairing unit
    for a matched-pairs design (see prereg_addendum_B_sham_regulation.md
    \u00a73/\u00a75). Returns (pairs, n_a_unpaired) where pairs is a list of
    (success_a, success_b) booleans, and n_a_unpaired counts condition_a
    rows that had no matching condition_b row (expected to be 0 or very
    small -- see the addendum's \u00a75 for why this can happen and why it's
    harmless).
    """
    cell = filter_rows(rows, agent_type=agent_type, max_steps=max_steps)
    a_rows = filter_rows(cell, condition=condition_a)
    b_by_key = {
        (r["seed"], r["task_id"]): r
        for r in filter_rows(cell, condition=condition_b)
    }

    pairs = []
    n_unpaired = 0
    for a_row in a_rows:
        key = (a_row["seed"], a_row["task_id"])
        b_row = b_by_key.get(key)
        if b_row is None:
            n_unpaired += 1
            continue
        pairs.append((a_row["success"], b_row["success"]))

    return pairs, n_unpaired


def mcnemar_discordant_counts(pairs):
    """
    From a list of (success_a, success_b) pairs, returns (a, b, c, d) per
    the 2x2 layout in prereg_addendum_B_sham_regulation.md \u00a73:
      a = both correct, b = a correct/b incorrect (discordant),
      c = a incorrect/b correct (discordant), d = both incorrect.
    """
    a = sum(1 for sa, sb in pairs if sa and sb)
    b = sum(1 for sa, sb in pairs if sa and not sb)
    c = sum(1 for sa, sb in pairs if (not sa) and sb)
    d = sum(1 for sa, sb in pairs if (not sa) and (not sb))
    return a, b, c, d


def mcnemar_test(b, c, exact_threshold=25):
    """
    McNemar's test on the discordant pair counts b, c (see
    mcnemar_discordant_counts). Returns
    (chi2, p_chi2, used_exact, exact_p_or_None), mirroring
    two_proportion_ztest's return shape for consistency.

    Continuity-corrected chi-square if b+c >= exact_threshold; exact
    binomial two-sided test otherwise (per
    prereg_addendum_B_sham_regulation.md \u00a73's b+c<25 fallback).
    """
    n_discordant = b + c

    if n_discordant == 0:
        # No discordant pairs at all -- perfectly concordant, p=1 by
        # construction; chi2 undefined, report as 0.
        return 0.0, 1.0, False, None

    used_exact = n_discordant < exact_threshold
    exact_p = None
    if used_exact:
        k = min(b, c)
        # Two-sided exact binomial test under H0: b, c ~ Binomial(b+c, 0.5)
        exact_p = 2 * stats.binom.cdf(k, n_discordant, 0.5)
        exact_p = min(exact_p, 1.0)

    chi2 = (abs(b - c) - 1) ** 2 / n_discordant
    p_chi2 = 1 - stats.chi2.cdf(chi2, df=1)

    return chi2, p_chi2, used_exact, exact_p


def paired_risk_difference_ci(b, c, n, alpha=0.05):
    """
    Normal-approximation CI for the paired risk difference
    d = (b - c) / n, per prereg_addendum_B_sham_regulation.md \u00a73:
      SE = sqrt[(b+c) - (b-c)^2/n] / n
    Returns (diff, lo, hi). n is the number of PAIRED observations (not
    b+c -- concordant pairs contribute to n but not to the point estimate
    or SE beyond that).
    """
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))

    diff = (b - c) / n
    inner = (b + c) - ((b - c) ** 2) / n
    inner = max(inner, 0.0)  # guard tiny negative floating-point noise
    se = math.sqrt(inner) / n
    z = stats.norm.ppf(1 - alpha / 2)
    return diff, diff - z * se, diff + z * se


# ----------------------------------------------------------------------
# Co-primary endpoints
# ----------------------------------------------------------------------

def compute_independent_endpoint(rows, endpoint_id, condition_a, condition_b,
                                  agent_type, max_steps, alpha):
    """
    H1 or H1-B: independent-samples two-proportion z-test (Fisher's exact
    fallback), Newcombe risk-difference CI, log risk-ratio CI. Used for
    any pair that is NOT the paired governed/sham_regulation design.
    """
    cell = filter_rows(rows, agent_type=agent_type, max_steps=max_steps)
    a_rows = filter_rows(cell, condition=condition_a)
    b_rows = filter_rows(cell, condition=condition_b)

    n1, n2 = len(a_rows), len(b_rows)
    if n1 == 0 or n2 == 0:
        raise ValueError(
            f"[{endpoint_id}] endpoint cell is empty (n_{condition_a}={n1}, "
            f"n_{condition_b}={n2}). Check agent_type='{agent_type}' and "
            f"max_steps={max_steps} exist in the input CSV."
        )

    x1 = sum(1 for r in a_rows if r["success"])
    x2 = sum(1 for r in b_rows if r["success"])

    z, p_z, used_exact, p_exact = two_proportion_ztest(x1, n1, x2, n2)
    reported_p = p_exact if used_exact else p_z

    diff, diff_lo, diff_hi = newcombe_risk_difference_ci(x1, n1, x2, n2, alpha)
    rr, rr_lo, rr_hi = log_risk_ratio_ci(x1, n1, x2, n2, alpha)
    p1_lo, p1_hi = wilson_ci(x1, n1, alpha)
    p2_lo, p2_hi = wilson_ci(x2, n2, alpha)

    return {
        "spec": {
            "endpoint_id": endpoint_id,
            "design": "independent_samples",
            "agent_type": agent_type,
            "max_steps": max_steps,
            "condition_a": condition_a,
            "condition_b": condition_b,
            "alpha": alpha,
            "test": "two-sided two-proportion z-test (pooled variance); "
                    "Fisher's exact substituted for the p-value if any cell < 5",
        },
        "n_a": n1, "successes_a": x1, "success_rate_a": x1 / n1,
        "success_rate_a_wilson_ci": [p1_lo, p1_hi],
        "n_b": n2, "successes_b": x2, "success_rate_b": x2 / n2,
        "success_rate_b_wilson_ci": [p2_lo, p2_hi],
        "risk_difference_a_minus_b": diff,
        "risk_difference_95ci_newcombe": [diff_lo, diff_hi],
        "risk_ratio_a_over_b": rr,
        "risk_ratio_95ci_log_normal": [rr_lo, rr_hi],
        "z_statistic": z,
        "p_value_z_test": p_z,
        "used_fisher_exact": used_exact,
        "p_value_fisher_exact": p_exact,
        "reported_p_value": reported_p,
        "reject_h0": reported_p < alpha,
    }


def compute_paired_endpoint(rows, endpoint_id, condition_a, condition_b,
                             agent_type, max_steps, alpha):
    """
    H1-C: McNemar's test on governed/sham_regulation pairs matched by
    (seed, task_id). See module docstring and
    prereg_addendum_B_sham_regulation.md \u00a73 for why this must differ from
    compute_independent_endpoint.
    """
    pairs, n_unpaired = pair_by_seed_task(rows, agent_type, max_steps, condition_a, condition_b)
    n = len(pairs)

    if n == 0:
        raise ValueError(
            f"[{endpoint_id}] no paired observations found for "
            f"{condition_a} vs {condition_b} at agent_type='{agent_type}', "
            f"max_steps={max_steps}."
        )

    a, b, c, d = mcnemar_discordant_counts(pairs)
    chi2, p_chi2, used_exact, p_exact = mcnemar_test(b, c)
    reported_p = p_exact if used_exact else p_chi2

    diff, diff_lo, diff_hi = paired_risk_difference_ci(b, c, n, alpha)

    x_a = sum(1 for sa, _ in pairs if sa)
    x_b = sum(1 for _, sb in pairs if sb)

    return {
        "spec": {
            "endpoint_id": endpoint_id,
            "design": "matched_pairs",
            "agent_type": agent_type,
            "max_steps": max_steps,
            "condition_a": condition_a,
            "condition_b": condition_b,
            "alpha": alpha,
            "test": "McNemar's test (continuity-corrected chi-square, df=1); "
                    "exact binomial substituted if discordant pairs (b+c) < 25",
        },
        "n_pairs": n,
        "n_a_unpaired_dropped": n_unpaired,
        "success_rate_a": x_a / n,
        "success_rate_b": x_b / n,
        "both_correct": a, "a_correct_b_incorrect": b,
        "a_incorrect_b_correct": c, "both_incorrect": d,
        "n_discordant": b + c,
        "risk_difference_a_minus_b": diff,
        "risk_difference_95ci_normal_approx": [diff_lo, diff_hi],
        "mcnemar_chi2": chi2,
        "p_value_chi2": p_chi2,
        "used_exact_binomial": used_exact,
        "p_value_exact_binomial": p_exact,
        "reported_p_value": reported_p,
        "reject_h0": reported_p < alpha,
    }


def mh_robustness_check(rows):
    """
    Stratified-by-task Cochran-Mantel-Haenszel check on H1
    (governed-vs-static_good, persistent/16-steps). Not part of the
    confirmatory decision for ANY endpoint; reported alongside H1 to check
    the pooled-across-tasks z-test isn't hiding Simpson's paradox
    (different tasks may have very different baseline difficulty).
    """
    cell = filter_rows(rows, agent_type=PRIMARY_AGENT_TYPE, max_steps=PRIMARY_MAX_STEPS)
    task_ids = sorted(set(r["task_id"] for r in cell))

    strata = []
    per_task_rates = []
    for task_id in task_ids:
        a_rows = filter_rows(cell, task_id=task_id, condition=H1_CONDITIONS[0])
        b_rows = filter_rows(cell, task_id=task_id, condition=H1_CONDITIONS[1])
        a = sum(1 for r in a_rows if r["success"])
        b = len(a_rows) - a
        c = sum(1 for r in b_rows if r["success"])
        d = len(b_rows) - c
        strata.append((a, b, c, d))
        per_task_rates.append({
            "task_id": task_id,
            "n_governed": len(a_rows),
            "success_rate_governed": a / len(a_rows) if a_rows else float("nan"),
            "n_static_good": len(b_rows),
            "success_rate_static_good": c / len(b_rows) if b_rows else float("nan"),
        })

    mh_or, chi2, p = cochran_mantel_haenszel(strata)
    return {
        "mh_common_odds_ratio": mh_or,
        "cmh_chi2_statistic": chi2,
        "cmh_p_value": p,
        "n_strata": len(strata),
        "per_task_rates": per_task_rates,
        "note": "Robustness check only, stratified by task_id, for H1 "
                "(governed vs static_good). Not the primary confirmatory "
                "test for any endpoint (see prereg_cas_governor.md \u00a73/\u00a77).",
    }


# ----------------------------------------------------------------------
# Secondary endpoints
# ----------------------------------------------------------------------

def secondary_endpoints(rows):
    records = []
    for agent_type, max_steps, (cond_a, cond_b) in product(
        ALL_AGENT_TYPES, ALL_STEP_BUDGETS, SECONDARY_COMPARISONS
    ):
        is_primary_cell = (agent_type, max_steps) == (PRIMARY_AGENT_TYPE, PRIMARY_MAX_STEPS)
        pair = (cond_a, cond_b)

        # H1-B's and H1-C's primary cell already has its own dedicated
        # co-primary result -- don't also count it in the exploratory pool.
        if is_primary_cell and pair in (H1B_CONDITIONS, H1C_CONDITIONS):
            continue

        if pair == H1C_CONDITIONS:
            # Matched-pairs design -- McNemar, not the independent z-test.
            pairs, n_unpaired = pair_by_seed_task(rows, agent_type, max_steps, cond_a, cond_b)
            n = len(pairs)
            if n == 0:
                continue

            a, b, c, d = mcnemar_discordant_counts(pairs)
            chi2, p_chi2, used_exact, p_exact = mcnemar_test(b, c)
            reported_p = p_exact if used_exact else p_chi2
            diff, diff_lo, diff_hi = paired_risk_difference_ci(b, c, n, PRIMARY_ALPHA)
            x_a = sum(1 for sa, _ in pairs if sa)
            x_b = sum(1 for _, sb in pairs if sb)

            records.append({
                "agent_type": agent_type,
                "max_steps": max_steps,
                "condition_a": cond_a,
                "condition_b": cond_b,
                "test_type": "mcnemar_paired",
                "n_a": n,
                "success_rate_a": x_a / n,
                "n_b": n,
                "success_rate_b": x_b / n,
                "risk_difference": diff,
                "risk_difference_lo": diff_lo,
                "risk_difference_hi": diff_hi,
                "z_statistic": None,
                "mcnemar_chi2": chi2,
                "raw_p_value": reported_p,
                "used_fisher_exact": False,
                "used_exact_binomial": used_exact,
            })
            continue

        # Independent-samples path (everything except H1-C).
        cell = filter_rows(rows, agent_type=agent_type, max_steps=max_steps)
        a_rows = filter_rows(cell, condition=cond_a)
        b_rows = filter_rows(cell, condition=cond_b)

        n1, n2 = len(a_rows), len(b_rows)
        if n1 == 0 or n2 == 0:
            continue

        x1 = sum(1 for r in a_rows if r["success"])
        x2 = sum(1 for r in b_rows if r["success"])

        z, p_z, used_exact, p_exact = two_proportion_ztest(x1, n1, x2, n2)
        reported_p = p_exact if used_exact else p_z
        diff, diff_lo, diff_hi = newcombe_risk_difference_ci(x1, n1, x2, n2, PRIMARY_ALPHA)

        records.append({
            "agent_type": agent_type,
            "max_steps": max_steps,
            "condition_a": cond_a,
            "condition_b": cond_b,
            "test_type": "independent_z",
            "n_a": n1,
            "success_rate_a": x1 / n1,
            "n_b": n2,
            "success_rate_b": x2 / n2,
            "risk_difference": diff,
            "risk_difference_lo": diff_lo,
            "risk_difference_hi": diff_hi,
            "z_statistic": z,
            "mcnemar_chi2": None,
            "raw_p_value": reported_p,
            "used_fisher_exact": used_exact,
            "used_exact_binomial": False,
        })

    # ONE joint BH-FDR pool across every row, independent_z and
    # mcnemar_paired alike -- per prereg_addendum_B_sham_regulation.md \u00a76,
    # sham's secondary cells join the SAME family, not a separate one.
    pvals = [r["raw_p_value"] for r in records]
    adjusted, reject = benjamini_hochberg(pvals, SECONDARY_ALPHA_FDR)
    for r, adj_p, rej in zip(records, adjusted, reject):
        r["bh_adjusted_p_value"] = adj_p
        r[f"reject_at_fdr_{SECONDARY_ALPHA_FDR}"] = rej

    return records


# ----------------------------------------------------------------------
# Descriptive outputs
# ----------------------------------------------------------------------

def task_breakdown_primary_cell(rows):
    cell = filter_rows(rows, agent_type=PRIMARY_AGENT_TYPE, max_steps=PRIMARY_MAX_STEPS)
    task_ids = sorted(set(r["task_id"] for r in cell))
    out = []
    for task_id in task_ids:
        row = {"task_id": task_id}
        for cond in ALL_CONDITIONS:
            sub = filter_rows(cell, task_id=task_id, condition=cond)
            n = len(sub)
            succ = sum(1 for r in sub if r["success"])
            row[f"{cond}_n"] = n
            row[f"{cond}_success_rate"] = succ / n if n else float("nan")
        out.append(row)
    return out


def descriptive_grid_summary(rows):
    out = []
    for agent_type, max_steps, condition in product(ALL_AGENT_TYPES, ALL_STEP_BUDGETS, ALL_CONDITIONS):
        sub = filter_rows(rows, agent_type=agent_type, max_steps=max_steps, condition=condition)
        n = len(sub)
        if n == 0:
            continue

        def mean_of(field):
            return sum(r[field] for r in sub) / n

        out.append({
            "agent_type": agent_type,
            "max_steps": max_steps,
            "condition": condition,
            "n": n,
            "success_rate": sum(1 for r in sub if r["success"]) / n,
            "answered_rate": sum(1 for r in sub if r["answered"]) / n,
            "mean_total_tool_calls": mean_of("total_tool_calls"),
            "mean_failed_tool_calls": mean_of("failed_tool_calls"),
            "mean_repeated_failed_actions": mean_of("repeated_failed_actions"),
            "mean_post_error_tool_burden": mean_of("post_error_tool_burden"),
            "mean_regulatory_signals": mean_of("regulatory_signals"),
            "mean_recovering_signals": mean_of("recovering_signals"),
            "mean_stuck_signals": mean_of("stuck_signals"),
            "mean_soft_warnings": mean_of("soft_warnings"),
            "mean_hard_blocks": mean_of("hard_blocks"),
        })
    return out


# ----------------------------------------------------------------------
# Output writing
# ----------------------------------------------------------------------

def _to_native(obj):
    """
    Recursively convert numpy scalar types (numpy.float64, numpy.bool_,
    numpy.int64, etc. -- which scipy.stats functions return, and which
    the stdlib json module does not know how to serialize) into native
    Python types, leaving dicts/lists/tuples structurally intact.
    """
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def write_csv(path, records):
    records = [_to_native(r) for r in records]
    if not records:
        with open(path, "w") as f:
            f.write("")
        return
    fieldnames = list(records[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow(r)


def _format_independent_block(title, result):
    r = result
    lines = [title, "-" * len(title)]
    lines.append(f"design: independent samples (two-proportion z-test)")
    lines.append(f"agent_type = {r['spec']['agent_type']}, max_steps = {r['spec']['max_steps']}")
    lines.append(f"{r['spec']['condition_a']}: n={r['n_a']}, success_rate={r['success_rate_a']:.4f} "
                 f"(95% Wilson CI {r['success_rate_a_wilson_ci'][0]:.4f}-{r['success_rate_a_wilson_ci'][1]:.4f})")
    lines.append(f"{r['spec']['condition_b']}: n={r['n_b']}, success_rate={r['success_rate_b']:.4f} "
                 f"(95% Wilson CI {r['success_rate_b_wilson_ci'][0]:.4f}-{r['success_rate_b_wilson_ci'][1]:.4f})")
    lines.append(f"Risk difference (a - b): {r['risk_difference_a_minus_b']:.4f} "
                 f"(95% CI {r['risk_difference_95ci_newcombe'][0]:.4f}-{r['risk_difference_95ci_newcombe'][1]:.4f})")
    lines.append(f"Risk ratio (a / b): {r['risk_ratio_a_over_b']:.4f} "
                 f"(95% CI {r['risk_ratio_95ci_log_normal'][0]:.4f}-{r['risk_ratio_95ci_log_normal'][1]:.4f})")
    if r["used_fisher_exact"]:
        lines.append(f"Small-cell count triggered Fisher's exact test: p = {r['p_value_fisher_exact']:.6f}")
    else:
        lines.append(f"z = {r['z_statistic']:.4f}, two-sided p = {r['p_value_z_test']:.6f}")
    lines.append(f"Reported p-value: {r['reported_p_value']:.6f}  (alpha = {r['spec']['alpha']})")
    lines.append(f"Reject H0: {r['reject_h0']}")
    return lines


def _format_paired_block(title, result):
    r = result
    lines = [title, "-" * len(title)]
    lines.append("design: MATCHED PAIRS (McNemar's test) -- see module docstring for why")
    lines.append(f"agent_type = {r['spec']['agent_type']}, max_steps = {r['spec']['max_steps']}")
    lines.append(f"n paired observations = {r['n_pairs']}  "
                 f"({r['spec']['condition_a']} rows dropped for no matching pair: {r['n_a_unpaired_dropped']})")
    lines.append(f"{r['spec']['condition_a']} success rate: {r['success_rate_a']:.4f}   "
                 f"{r['spec']['condition_b']} success rate: {r['success_rate_b']:.4f}")
    lines.append(f"2x2: both_correct={r['both_correct']}, "
                 f"{r['spec']['condition_a']}_only_correct={r['a_correct_b_incorrect']}, "
                 f"{r['spec']['condition_b']}_only_correct={r['a_incorrect_b_correct']}, "
                 f"both_incorrect={r['both_incorrect']}  (discordant total = {r['n_discordant']})")
    lines.append(f"Paired risk difference (a - b): {r['risk_difference_a_minus_b']:.4f} "
                 f"(95% CI {r['risk_difference_95ci_normal_approx'][0]:.4f}-{r['risk_difference_95ci_normal_approx'][1]:.4f})")
    if r["used_exact_binomial"]:
        lines.append(f"Discordant pairs < 25: exact binomial test: p = {r['p_value_exact_binomial']:.6f}")
    else:
        lines.append(f"McNemar chi2 (continuity-corrected) = {r['mcnemar_chi2']:.4f}, p = {r['p_value_chi2']:.6f}")
    lines.append(f"Reported p-value: {r['reported_p_value']:.6f}  (alpha = {r['spec']['alpha']})")
    lines.append(f"Reject H0: {r['reject_h0']}")
    return lines


def write_primary_report(path_txt, path_json, h1_result, h1b_result, h1c_result, mh_result):
    with open(path_json, "w", encoding="utf-8") as f:
        json.dump(
            _to_native({
                "H1_governed_vs_static_good": h1_result,
                "H1B_governed_vs_disturbance_first": h1b_result,
                "H1C_governed_vs_sham_regulation": h1c_result,
                "mh_robustness_check_H1": mh_result,
            }),
            f,
            indent=2,
        )

    lines = []
    lines.append("CO-PRIMARY ENDPOINTS (pre-registered, each run once)")
    lines.append("=" * 72)
    lines.append("")
    lines += _format_independent_block(
        "H1: governed vs static_good  (prereg_cas_governor.md)", h1_result)
    lines.append("")
    lines += _format_independent_block(
        "H1-B: governed vs disturbance_first  (Addendum A)", h1b_result)
    lines.append("")
    lines += _format_paired_block(
        "H1-C: governed vs sham_regulation  (Addendum B)", h1c_result)
    lines.append("")
    lines.append("=" * 72)
    lines.append("ROBUSTNESS CHECK for H1 (Cochran-Mantel-Haenszel, stratified by task_id)")
    lines.append("Not a confirmatory test for any endpoint -- see prereg \u00a75/\u00a77.")
    lines.append(f"MH common odds ratio: {mh_result['mh_common_odds_ratio']:.4f}")
    lines.append(f"CMH chi2 = {mh_result['cmh_chi2_statistic']:.4f}, p = {mh_result['cmh_p_value']:.6f}, "
                 f"strata (tasks) = {mh_result['n_strata']}")
    lines.append("")
    lines.append("Per-task success rates for H1 (descriptive):")
    for row in mh_result["per_task_rates"]:
        lines.append(
            f"  {row['task_id']:35s} governed={row['success_rate_governed']:.3f} "
            f"(n={row['n_governed']:5d})   static_good={row['success_rate_static_good']:.3f} "
            f"(n={row['n_static_good']:5d})"
        )

    with open(path_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("\n".join(lines))


# ----------------------------------------------------------------------
# Self-test: tiny synthetic fixture, exercises every code path without
# requiring the Java experiment to have been run. Does not touch or
# validate any scientific claim -- purely checks the script itself runs.
# ----------------------------------------------------------------------

def _write_synthetic_fixture(path):
    header = (
        "seed,task_id,agent_type,condition,max_steps,agent,total_tool_calls,"
        "failed_tool_calls,successful_tool_calls,repeated_failed_actions,"
        "repeated_successful_actions,governor_interventions,regulatory_signals,"
        "disturbed_signals,recovering_signals,stuck_signals,released_signals,"
        "answered,success,steps_after_first_failure,post_error_tool_burden,"
        "soft_warnings,hard_blocks"
    )
    rng = np.random.default_rng(42)
    rows = [header]

    task_ids = [f"T{i:03d}_synthetic" for i in range(1, 9)]
    base_rate = {"governed": 0.55, "static_good": 0.48, "ungoverned": 0.30,
                 "reweight_only": 0.45, "block_only": 0.40, "disturbance_first": 0.50,
                 "sham_regulation": 0.33}

    for agent_type in ALL_AGENT_TYPES:
        for max_steps in ALL_STEP_BUDGETS:
            for condition in ALL_CONDITIONS:
                for task_id in task_ids:
                    n_seeds = 50  # small, just to exercise the code
                    p = base_rate[condition]
                    for seed in range(n_seeds):
                        success = rng.random() < p
                        answered = success or (rng.random() < 0.5)
                        row = [
                            seed, task_id, agent_type, condition, max_steps,
                            f"{agent_type}_{condition}_steps{max_steps}",
                            rng.integers(1, 10), rng.integers(0, 5), rng.integers(1, 8),
                            rng.integers(0, 3), rng.integers(0, 3), rng.integers(0, 2),
                            rng.integers(0, 6), rng.integers(0, 3), rng.integers(0, 3),
                            rng.integers(0, 2), rng.integers(0, 2),
                            str(answered).lower(), str(success).lower(),
                            rng.integers(0, 5), rng.integers(0, 8),
                            rng.integers(0, 2), rng.integers(0, 2),
                        ]
                        rows.append(",".join(str(x) for x in row))

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(rows) + "\n")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default="output/stochastic_policy_runs.csv")
    parser.add_argument("--outdir", default="analysis_output")
    parser.add_argument("--self-test", action="store_true",
                         help="Run against a small synthetic fixture instead of real data, "
                              "to smoke-test the script. Does not produce a valid scientific result.")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    input_path = args.input
    if args.self_test:
        input_path = os.path.join(args.outdir, "_synthetic_fixture.csv")
        print(f"[--self-test] Writing synthetic fixture to {input_path} ...")
        _write_synthetic_fixture(input_path)
        print("[--self-test] NOTE: results below are from fake random data, not a real "
              "experiment run. This only checks the script executes correctly end-to-end.")

    if not os.path.exists(input_path):
        print(f"ERROR: input file not found: {input_path}", file=sys.stderr)
        print("Run RunStochasticPolicyExperiment.java first (it writes "
              "output/stochastic_policy_runs.csv), or pass --self-test to smoke-test "
              "this script against synthetic data.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {input_path} ...")
    rows = load_data(input_path)
    print(f"Loaded {len(rows)} rows.")

    print("\nComputing three co-primary endpoints (each run once) ...")
    h1_result = compute_independent_endpoint(
        rows, "H1", H1_CONDITIONS[0], H1_CONDITIONS[1],
        PRIMARY_AGENT_TYPE, PRIMARY_MAX_STEPS, PRIMARY_ALPHA
    )
    h1b_result = compute_independent_endpoint(
        rows, "H1-B", H1B_CONDITIONS[0], H1B_CONDITIONS[1],
        PRIMARY_AGENT_TYPE, PRIMARY_MAX_STEPS, PRIMARY_ALPHA
    )
    h1c_result = compute_paired_endpoint(
        rows, "H1-C", H1C_CONDITIONS[0], H1C_CONDITIONS[1],
        PRIMARY_AGENT_TYPE, PRIMARY_MAX_STEPS, PRIMARY_ALPHA
    )
    mh_result = mh_robustness_check(rows)

    write_primary_report(
        os.path.join(args.outdir, "primary_endpoint_result.txt"),
        os.path.join(args.outdir, "primary_endpoint_result.json"),
        h1_result, h1b_result, h1c_result, mh_result,
    )

    print("\nComputing secondary endpoints with BH-FDR correction ...")
    secondary_records = secondary_endpoints(rows)
    write_csv(os.path.join(args.outdir, "secondary_endpoint_results.csv"), secondary_records)
    n_reject = sum(1 for r in secondary_records if r[f"reject_at_fdr_{SECONDARY_ALPHA_FDR}"])
    print(f"  {len(secondary_records)} secondary tests computed "
          f"({sum(1 for r in secondary_records if r['test_type']=='mcnemar_paired')} paired/McNemar, "
          f"{sum(1 for r in secondary_records if r['test_type']=='independent_z')} independent-samples); "
          f"{n_reject} reject H0 at FDR={SECONDARY_ALPHA_FDR}.")

    print("\nComputing descriptive per-task breakdown for the primary cell ...")
    task_records = task_breakdown_primary_cell(rows)
    write_csv(os.path.join(args.outdir, "task_breakdown_primary_cell.csv"), task_records)

    print("Computing full descriptive grid ...")
    grid_records = descriptive_grid_summary(rows)
    write_csv(os.path.join(args.outdir, "descriptive_grid_summary.csv"), grid_records)

    print(f"\nAll outputs written to {args.outdir}/")
    print("  - primary_endpoint_result.txt / .json  (H1, H1-B, H1-C + MH robustness check)")
    print("  - secondary_endpoint_results.csv")
    print("  - task_breakdown_primary_cell.csv")
    print("  - descriptive_grid_summary.csv")


if __name__ == "__main__":
    main()
