"""
analyze_switching.py

Tests paper 3's core prediction on the discrete tool-use substrate: does
a switching agent (alternating RF/DF dwell blocks within one episode)
require LESS regulatory burden than an occupancy-weighted mixture of
fixed-RF (governed) and fixed-DF (disturbance_first) trajectories would
predict?

C_switch = actual_switching_metric - [p_RF * metric_RF_fixed + p_DF * metric_DF_fixed]

A reliably NEGATIVE C_switch across dwell lengths replicates paper 3's
"Intermittent Control Is Not Diluted Control" finding. A reliably
POSITIVE or null result means the effect does not generalize to this
substrate -- report whichever the data actually shows.

Two regulatory-burden proxies are used (no direct continuous mu exists on
this discrete substrate): mean_hard_blocks (how often the governor had to
intervene) and mean_post_error_tool_burden (accumulated post-failure
burden) -- paper 3 itself uses multiple complementary diagnostics rather
than a single number, so this follows the same practice.

Uncertainty (SEM propagation)
-----------------------------
Writing the per-cell statistic as

    C_switch = A - P*R - (1-P)*D

with A = mean actual switching metric, P = mean realized RF occupancy
(both from the switching runs), and R, D = the fixed-mode reference means
(governed / disturbance_first), C_switch has TWO independent sources of
sampling error, and an earlier version of this script accounted for only
the first:

  1. Switching-sampling error. The standard error of the mean of the
     per-episode quantity g_i = actual_i - (p_rf_i*R + (1-p_rf_i)*D),
     treating R and D as fixed. This is identical to what a one-sample
     t-test on g_i reports, and it already folds in the joint variability
     of actual_i and p_rf_i (they share an episode, so they may covary).

  2. Reference-sampling error. R and D are themselves estimated from
     ~16000 fixed-mode episodes each, with their own SEMs. Propagating
     them through C_switch (partials -P and -(1-P), the two fixed sets
     independent of each other and of the switching runs):

         SE_ref = sqrt( P^2 * SE_R^2 + (1-P)^2 * SE_D^2 )

The two sources are independent, so SE_total = sqrt(SE_switch^2 + SE_ref^2).
The old t-test used SE_switch alone, which OVERSTATES significance -- at
n=16000 it reported p=0.0000 for effects of ~0.04. z = C_switch/SE_total
and the 95% CI = C_switch +- 1.96*SE_total use the honest denominator; the
normal approximation is used because n is large.

Reference SEMs are taken from the per-episode fixed-mode runs file when it
is supplied (recommended -- the aggregate has no sem_hard_blocks column),
otherwise from the aggregate's sem_<metric> columns where they exist. If a
metric's reference SEM cannot be found, its CI falls back to the
switching-only SE and is flagged.

Usage:
    python3 analyze_switching.py switching_runs.csv stochastic_policy_aggregate.csv \\
        [stochastic_policy_runs.csv]
"""

import sys
import numpy as np
import pandas as pd
from scipy import stats

Z_95 = 1.959963984540054  # two-sided 95% normal quantile


def fixed_reference_ses(existing, fixed_runs, metrics):
    """Map (agent_type, max_steps, condition) -> {metric: SEM of that mean}.

    Prefers the per-episode fixed-mode runs (SEM computed directly, available
    for every metric); falls back to the aggregate's sem_<metric> columns,
    which do NOT include hard_blocks -- missing entries come back as NaN so
    the caller can flag the affected cells rather than silently drop the
    reference-error term.
    """
    ses = {}
    if fixed_runs is not None:
        for (agent_type, condition, max_steps), grp in fixed_runs.groupby(
            ["agent_type", "condition", "max_steps"]
        ):
            n = len(grp)
            ses[(agent_type, max_steps, condition)] = {
                m: grp[m].std(ddof=1) / np.sqrt(n) for m in metrics
            }
    else:
        for _, row in existing.iterrows():
            ses[(row["agent_type"], row["max_steps"], row["condition"])] = {
                m: (row[f"sem_{m}"] if f"sem_{m}" in existing.columns else np.nan)
                for m in metrics
            }
    return ses


def main():
    if len(sys.argv) not in (3, 4):
        print("Usage: python3 analyze_switching.py switching_runs.csv "
              "stochastic_policy_aggregate.csv [stochastic_policy_runs.csv]")
        sys.exit(1)

    switching_path, existing_agg_path = sys.argv[1], sys.argv[2]
    fixed_runs_path = sys.argv[3] if len(sys.argv) == 4 else None

    switching = pd.read_csv(switching_path)
    existing = pd.read_csv(existing_agg_path)
    fixed_runs = pd.read_csv(fixed_runs_path) if fixed_runs_path else None

    if fixed_runs is None:
        print("NOTE: no fixed-mode runs file given; reference SEMs come from the "
              "aggregate's sem_<metric> columns. hard_blocks has no such column, "
              "so its reference-error term will be missing. Pass "
              "stochastic_policy_runs.csv as a third argument for full propagation.\n")

    # Per-episode realized RF occupancy (NOT assumed 0.5 -- episodes that
    # end early, e.g. a final answer accepted mid-dwell-block, can have
    # off-schedule realized occupancy; this uses the actual step counts).
    switching["p_rf"] = switching["rf_steps"] / (switching["rf_steps"] + switching["df_steps"])

    fixed_governed = existing[existing["condition"] == "governed"].set_index(["agent_type", "max_steps"])
    fixed_df = existing[existing["condition"] == "disturbance_first"].set_index(["agent_type", "max_steps"])

    burden_metrics = ["hard_blocks", "post_error_tool_burden"]

    ref_ses = fixed_reference_ses(existing, fixed_runs, burden_metrics)
    missing_ref_se = False

    results = []

    for (agent_type, dwell_length, max_steps), group in switching.groupby(
        ["agent_type", "dwell_length", "max_steps"]
    ):
        key = (agent_type, max_steps)
        if key not in fixed_governed.index or key not in fixed_df.index:
            print(f"WARNING: no matching fixed-mode reference for {key}, skipping.")
            continue

        rf_row = fixed_governed.loc[key]
        df_row = fixed_df.loc[key]
        mean_p_rf = group["p_rf"].mean()

        for metric in burden_metrics:
            rf_fixed_mean = rf_row[f"mean_{metric}"]
            df_fixed_mean = df_row[f"mean_{metric}"]

            # Per-episode occupancy-weighted prediction, then compare to
            # this episode's own actual value -- gives a real distribution
            # to test, not just one pooled number.
            predicted_per_episode = group["p_rf"] * rf_fixed_mean + (1 - group["p_rf"]) * df_fixed_mean
            actual_per_episode = group[metric]
            c_switch = actual_per_episode - predicted_per_episode
            n = len(group)

            # Source 1: switching-sampling SE -- SE of the mean of the
            # per-episode c_switch (R, D held fixed). Identical to the
            # denominator of the old one-sample t-test.
            se_switch = c_switch.std(ddof=1) / np.sqrt(n)

            # Source 2: reference-sampling SE -- R and D are estimates.
            # Partials are -P and -(1-P); the two fixed sets are independent.
            se_rf = ref_ses.get((agent_type, max_steps, "governed"), {}).get(metric, np.nan)
            se_df = ref_ses.get((agent_type, max_steps, "disturbance_first"), {}).get(metric, np.nan)
            se_reference = np.sqrt(
                (mean_p_rf ** 2) * (se_rf ** 2)
                + ((1 - mean_p_rf) ** 2) * (se_df ** 2)
            )

            reference_se_missing = np.isnan(se_reference)
            if reference_se_missing:
                missing_ref_se = True
                # Fall back to switching-only SE so the row is still usable,
                # but the flag column below marks it as un-propagated.
                se_total = se_switch
            else:
                se_total = np.sqrt(se_switch ** 2 + se_reference ** 2)

            c_mean = c_switch.mean()
            z = c_mean / se_total if se_total > 0 else np.nan
            p = 2.0 * stats.norm.sf(abs(z)) if np.isfinite(z) else np.nan
            ci_lo = c_mean - Z_95 * se_total
            ci_hi = c_mean + Z_95 * se_total

            results.append({
                "agent_type": agent_type,
                "dwell_length": dwell_length,
                "max_steps": max_steps,
                "metric": metric,
                "n": n,
                "mean_p_rf": mean_p_rf,
                "rf_fixed_mean": rf_fixed_mean,
                "df_fixed_mean": df_fixed_mean,
                "occupancy_weighted_prediction": predicted_per_episode.mean(),
                "actual_switching_mean": actual_per_episode.mean(),
                "C_switch": c_mean,
                "se_switch": se_switch,
                "se_reference": se_reference,
                "se_total": se_total,
                "ci95_lo": ci_lo,
                "ci95_hi": ci_hi,
                "z": z,
                "p": p,
                "reference_se_missing": reference_se_missing,
            })

    results_df = pd.DataFrame(results)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)

    # A cell is "reliable" when its 95% CI excludes 0 -- equivalent to the
    # propagated two-sided p < 0.05, but stated as the interval so the
    # effect size and its uncertainty are both visible.
    ci_excludes_zero = (results_df["ci95_lo"] > 0) | (results_df["ci95_hi"] < 0)

    display_cols = [
        "agent_type", "dwell_length", "max_steps", "metric", "n", "mean_p_rf",
        "occupancy_weighted_prediction", "actual_switching_mean", "C_switch",
        "se_switch", "se_reference", "se_total", "ci95_lo", "ci95_hi", "z", "p",
    ]

    print("=" * 120)
    print("Switching-relief test: C_switch = actual - occupancy_weighted_prediction")
    print("Negative C_switch whose 95% CI excludes 0 replicates paper 3's regulatory-relief finding.")
    print("se_total = sqrt(se_switch^2 + se_reference^2); se_reference propagates the fixed-mode reference SEMs.")
    print("=" * 120)
    print(results_df[display_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    if missing_ref_se:
        n_missing = int(results_df["reference_se_missing"].sum())
        print(f"\n*** {n_missing} cell(s) lack a reference SEM (se_reference=NaN); their se_total falls back "
              "to switching-only SE and is NOT fully propagated. Pass stochastic_policy_runs.csv to fix. ***")

    print()
    print("=" * 120)
    print("Effect of honest propagation: how many cells change significance vs the switching-only SE?")
    print("=" * 120)
    ci_switch_only = (results_df["C_switch"].abs() - Z_95 * results_df["se_switch"]) > 0
    flipped = int((ci_excludes_zero != ci_switch_only).sum())
    print(f"{flipped}/{len(results_df)} cell(s) flip significance once the reference-mode SEM is included.")

    print()
    print("=" * 120)
    print("Summary: fraction of tested cells whose 95% CI is reliably NEGATIVE / POSITIVE")
    print("=" * 120)
    for metric in burden_metrics:
        sub = results_df[results_df["metric"] == metric]
        sub_excl = (sub["ci95_lo"] > 0) | (sub["ci95_hi"] < 0)
        reliably_negative = int(((sub["C_switch"] < 0) & sub_excl).sum())
        reliably_positive = int(((sub["C_switch"] > 0) & sub_excl).sum())
        unresolved = len(sub) - reliably_negative - reliably_positive
        print(f"{metric}: {reliably_negative}/{len(sub)} reliably negative, "
              f"{reliably_positive}/{len(sub)} reliably positive, {unresolved}/{len(sub)} CI includes 0")

    results_df.to_csv("switching_analysis_results.csv", index=False)
    print("\nWrote: switching_analysis_results.csv")


if __name__ == "__main__":
    main()
