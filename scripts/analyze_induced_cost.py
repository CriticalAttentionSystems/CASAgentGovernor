"""
analyze_induced_cost.py

Phase 1 analysis for the discrete test of paper 4's control-generated
disturbance claim ("Noise-Source Structure in IRAM-Omega-Q"): when the
governor's own tightening carries an induced cost, does the discrete
substrate reproduce paper 4's amplification -> stochastic transition band ->
suppression arc as the cost scale alpha grows?

The experiment (RunInducedCostExperiment) gives the governor's positive
tightening dg+ an induced-failure probability min(1, alpha*dg+^2), injected
as a TRANSIENT tool failure that then feeds back into the governor's state.
The impulsive archetype is the primary target (the one agent whose burden
INCREASES under governance in the main sweep); persistent is the specificity
control.

This script reports, as a function of alpha and separately per ordering
(RF = anticipatory lag 0, DF = reactive lag 1):

  1. Paper-4 diagnostic panels: mean induced failures (effective-noise
     analog), mean regulation intensity g (mean_mu analog), mean burden,
     and success rate.

  2. The DF-RF paired contrast in burden with a 95% CI at each alpha. Paper
     4's transition band shows up as this CI WIDENING where stochastic seeds
     split between regulatory regimes -- so the CI-width-vs-alpha curve is
     the headline, not the means.

  3. A per-seed bimodality test (Hartigan-style dip via a simple two-mode
     separation on per-episode regulation intensity g, plus the fraction of
     seeds in a high-clamp vs suppressed regime) at each alpha, to detect
     the seed-splitting directly.

Phase-1 scope: the governor is cost-BLIND (it always escalates on failure,
it cannot deliberately back off), so paper 4's graceful cost-aware
suppression branch is NOT expected here. What is testable is amplification
and the transition band, plus a runaway collapse in success. See the agent
class doc for the phase-2 cost-aware variant needed for the suppression arm.

Usage:
    python3 analyze_induced_cost.py output/induced_cost_runs.csv \\
        [output/induced_cost_aggregate.csv]

Writes: induced_cost_analysis.csv
"""

import sys
import numpy as np
import pandas as pd


def bimodality_summary(g_values):
    """Cheap two-regime separation on per-episode mean regulation intensity.

    Returns (bimodality_coefficient, frac_high, gap). The bimodality
    coefficient b = (skew^2 + 1) / kurtosis exceeds ~0.555 for a uniform or
    bimodal distribution; higher means more clearly split. frac_high is the
    fraction above the midpoint of the observed range, and gap is the
    normalized separation between the two putative cluster means. These are
    heuristics for flagging where seeds split, not a formal test.
    """
    g = np.asarray(g_values, dtype=float)
    n = len(g)
    if n < 4 or np.allclose(g, g[0]):
        return np.nan, np.nan, np.nan

    m = g.mean()
    s = g.std(ddof=1)
    if s == 0:
        return np.nan, np.nan, np.nan
    skew = np.mean(((g - m) / s) ** 3)
    kurt = np.mean(((g - m) / s) ** 4)  # non-excess
    bc = (skew ** 2 + 1.0) / kurt if kurt > 0 else np.nan

    mid = 0.5 * (g.min() + g.max())
    low = g[g <= mid]
    high = g[g > mid]
    frac_high = len(high) / n
    if len(low) > 0 and len(high) > 0:
        gap = (high.mean() - low.mean()) / (g.max() - g.min())
    else:
        gap = 0.0
    return bc, frac_high, gap


def main():
    if len(sys.argv) not in (2, 3):
        print("Usage: python3 analyze_induced_cost.py induced_cost_runs.csv "
              "[induced_cost_aggregate.csv]")
        sys.exit(1)

    runs = pd.read_csv(sys.argv[1])

    # Per-episode mean regulation intensity is written per run row already; if
    # an older runs file lacks it, fall back to hard_blocks as a clamp proxy.
    intensity_col = ("mean_regulation_intensity"
                     if "mean_regulation_intensity" in runs.columns else "hard_blocks")

    burden_col = "post_error_tool_burden"
    z95 = 1.959963984540054

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)

    out_rows = []

    for (agent, max_steps), grp in runs.groupby(["agent_type", "max_steps"]):
        print("=" * 110)
        print(f"agent={agent}  max_steps={max_steps}")
        print("=" * 110)

        # --- Panels: per ordering, mean diagnostics vs alpha ---
        panel = grp.groupby(["ordering", "alpha"]).agg(
            n=("seed", "size"),
            induced=("induced_failures", "mean"),
            g=(intensity_col, "mean"),
            burden=(burden_col, "mean"),
            success=("success", "mean"),
        ).reset_index()
        print("\nDiagnostic panels (paper-4 analogs):")
        print(panel.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

        # --- DF - RF paired burden contrast with 95% CI, per alpha ---
        # Pairing is by seed x task x max_steps: same seed index for RF and DF.
        rf = grp[grp["ordering"] == "RF"]
        df = grp[grp["ordering"] == "DF"]
        key = ["seed", "task_id", "max_steps", "alpha"]
        merged = rf.merge(df, on=key, suffixes=("_rf", "_df"))
        merged["contrast"] = merged[f"{burden_col}_df"] - merged[f"{burden_col}_rf"]

        print("\nDF - RF burden contrast (transition band = CI WIDTH growing with alpha):")
        contrast_rows = []
        for alpha, cg in merged.groupby("alpha"):
            c = cg["contrast"].to_numpy()
            n = len(c)
            mean_c = c.mean()
            se = c.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
            ci_lo, ci_hi = mean_c - z95 * se, mean_c + z95 * se
            contrast_rows.append({
                "alpha": alpha, "n": n, "contrast": mean_c,
                "se": se, "ci95_lo": ci_lo, "ci95_hi": ci_hi, "ci_width": 2 * z95 * se,
            })
        contrast_df = pd.DataFrame(contrast_rows)
        print(contrast_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

        # --- Per-seed bimodality of regulation intensity, per ordering x alpha ---
        print("\nPer-seed regulation-intensity bimodality (seed-splitting signature):")
        bimo_rows = []
        for (ordering, alpha), og in grp.groupby(["ordering", "alpha"]):
            bc, frac_high, gap = bimodality_summary(og[intensity_col].to_numpy())
            bimo_rows.append({
                "ordering": ordering, "alpha": alpha, "n": len(og),
                "bimodality_coef": bc, "frac_high_clamp": frac_high, "cluster_gap": gap,
            })
        bimo_df = pd.DataFrame(bimo_rows)
        print(bimo_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print()

        # --- Accumulate a tidy long-form output row set ---
        for _, r in panel.iterrows():
            out_rows.append({
                "agent_type": agent, "max_steps": max_steps, "ordering": r["ordering"],
                "alpha": r["alpha"], "metric": "panel",
                "n": r["n"], "mean_induced_failures": r["induced"],
                "mean_regulation_intensity": r["g"], "mean_burden": r["burden"],
                "success_rate": r["success"],
            })
        for _, r in contrast_df.iterrows():
            out_rows.append({
                "agent_type": agent, "max_steps": max_steps, "ordering": "DF-RF",
                "alpha": r["alpha"], "metric": "contrast",
                "n": r["n"], "contrast": r["contrast"], "ci95_lo": r["ci95_lo"],
                "ci95_hi": r["ci95_hi"], "ci_width": r["ci_width"],
            })
        for _, r in bimo_df.iterrows():
            out_rows.append({
                "agent_type": agent, "max_steps": max_steps, "ordering": r["ordering"],
                "alpha": r["alpha"], "metric": "bimodality",
                "n": r["n"], "bimodality_coef": r["bimodality_coef"],
                "frac_high_clamp": r["frac_high_clamp"], "cluster_gap": r["cluster_gap"],
            })

    # --- Verdict: does the transition band appear? (CI width non-monotone,
    #     peaking at intermediate alpha rather than growing/shrinking flatly) ---
    print("=" * 110)
    print("Transition-band check: DF-RF contrast CI width by alpha (a peak at "
          "intermediate alpha is paper 4's signature)")
    print("=" * 110)
    tidy = pd.DataFrame(out_rows)
    band = tidy[tidy["metric"] == "contrast"]
    for (agent, max_steps), g in band.groupby(["agent_type", "max_steps"]):
        g = g.sort_values("alpha")
        widths = g["ci_width"].to_numpy()
        alphas = g["alpha"].to_numpy()
        peak_idx = int(np.nanargmax(widths))
        interior_peak = 0 < peak_idx < len(widths) - 1
        verdict = ("PEAK at intermediate alpha (transition-band consistent)"
                   if interior_peak else "monotone / edge peak (no clear band)")
        print(f"  {agent:11s} steps={max_steps}: max CI width at alpha={alphas[peak_idx]:.2f} -> {verdict}")

    tidy.to_csv("induced_cost_analysis.csv", index=False)
    print("\nWrote: induced_cost_analysis.csv")


if __name__ == "__main__":
    main()
