"""
analyze_cost_aware.py

Phase 2 analysis: does a cost-AWARE governor (one that backs off when its own
tightening has been generating failures) reproduce paper 4's graceful
suppression and stochastic transition band that the cost-BLIND governor of
phase 1 lacked -- and does backing off RESCUE the task success the cost-blind
governor destroyed at high induced-cost scale alpha?

Compares governor modes (blind, aware_k1.5, aware_k3.0, aware_k6.0) on the
same alpha sweep, per agent x ordering. Reports, as a function of alpha:

  1. SUPPRESSION: mean regulation intensity g vs alpha per mode. Paper 4's
     suppression = g FALLS as alpha grows. Cost-blind keeps g pinned; a
     cost-aware governor should bend g downward.

  2. RESCUE: success rate vs alpha per mode. Does aware recover the success
     blind lost? (For an agent that genuinely needs governance, it may not --
     that is itself a reportable result.)

  3. TRANSITION BAND: per-seed bimodality of episode regulation intensity, and
     the across-seed SD of burden, vs alpha per mode. A band shows up as
     bimodality / SD PEAKING at intermediate alpha under the cost-aware modes
     (seeds splitting into a regulated basin and a suppressed basin), not under
     blind.

Usage:
    python3 analyze_cost_aware.py output/cost_aware_runs.csv

Writes: cost_aware_analysis.csv
"""

import sys
import numpy as np
import pandas as pd


def bimodality_coef(values):
    """Sarle's bimodality coefficient b = (skew^2 + 1)/kurtosis (non-excess).

    b > ~0.555 suggests a bimodal/uniform (split) distribution; higher = more
    clearly two-mode. A heuristic flag for seed-splitting, not a formal test.
    """
    v = np.asarray(values, dtype=float)
    n = len(v)
    if n < 4:
        return np.nan
    s = v.std(ddof=1)
    if s == 0:
        return np.nan
    m = v.mean()
    skew = np.mean(((v - m) / s) ** 3)
    kurt = np.mean(((v - m) / s) ** 4)
    return (skew ** 2 + 1.0) / kurt if kurt > 0 else np.nan


def peakiness(alphas, values):
    """Return (peak_alpha, is_interior_peak) for a value-vs-alpha curve."""
    v = np.asarray(values, dtype=float)
    if np.all(np.isnan(v)):
        return np.nan, False
    idx = int(np.nanargmax(v))
    return alphas[idx], (0 < idx < len(v) - 1)


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 analyze_cost_aware.py cost_aware_runs.csv")
        sys.exit(1)

    runs = pd.read_csv(sys.argv[1])
    intensity_col = "mean_regulation_intensity"
    burden_col = "post_error_tool_burden"

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", None)

    mode_order = ["blind", "aware_k1.5", "aware_k3.0", "aware_k6.0"]
    modes = [m for m in mode_order if m in runs["mode"].unique()]

    out_rows = []

    for (agent, max_steps), grp in runs.groupby(["agent_type", "max_steps"]):
        print("=" * 120)
        print(f"agent={agent}  max_steps={max_steps}")
        print("=" * 120)

        # Pool orderings for the mode-vs-alpha summaries (RF and DF both present);
        # the DF-RF split is reported separately below.
        summary = grp.groupby(["mode", "alpha"]).agg(
            n=("seed", "size"),
            g=(intensity_col, "mean"),
            success=("success", "mean"),
            burden=(burden_col, "mean"),
            induced=("induced_failures", "mean"),
            backoff=("backoff_steps", "mean"),
        ).reset_index()

        print("\n[1+2] Intensity g (suppression) and success (rescue) vs alpha, by mode:")
        pivot_g = summary.pivot(index="alpha", columns="mode", values="g")[modes]
        pivot_s = summary.pivot(index="alpha", columns="mode", values="success")[modes]
        print("  regulation intensity g (paper-4 suppression = falls with alpha):")
        print(pivot_g.to_string(float_format=lambda x: f"{x:.3f}"))
        print("\n  success rate (rescue = aware above blind at high alpha):")
        print(pivot_s.to_string(float_format=lambda x: f"{x:.3f}"))

        # [3] Transition band: per-seed bimodality + burden SD vs alpha, per mode.
        print("\n[3] Transition-band signatures per mode (bimodality of episode g; SD of burden):")
        band_rows = []
        for mode in modes:
            for alpha, ag in grp[grp["mode"] == mode].groupby("alpha"):
                band_rows.append({
                    "mode": mode, "alpha": alpha, "n": len(ag),
                    "bimodality_g": bimodality_coef(ag[intensity_col].to_numpy()),
                    "sd_burden": ag[burden_col].std(ddof=1),
                })
        band = pd.DataFrame(band_rows)
        piv_bimo = band.pivot(index="alpha", columns="mode", values="bimodality_g")[modes]
        piv_sd = band.pivot(index="alpha", columns="mode", values="sd_burden")[modes]
        print("  bimodality coefficient of episode regulation intensity:")
        print(piv_bimo.to_string(float_format=lambda x: f"{x:.3f}"))
        print("\n  across-seed SD of burden:")
        print(piv_sd.to_string(float_format=lambda x: f"{x:.3f}"))

        # Verdicts per mode.
        print("\n  verdicts:")
        alphas = sorted(grp["alpha"].unique())
        for mode in modes:
            g_curve = pivot_g[mode].reindex(alphas).to_numpy()
            g_drop = np.nanmax(g_curve) - g_curve[-1]
            suppression = "YES" if g_drop > 0.03 else "no"

            bimo_curve = piv_bimo[mode].reindex(alphas).to_numpy()
            pk_a, interior = peakiness(alphas, bimo_curve)
            band_flag = "interior-peak" if interior else "monotone/edge"

            print(f"    {mode:11s} suppression(g drop={g_drop:+.3f})={suppression:3s}  "
                  f"bimodality peak@alpha={pk_a} ({band_flag})")

        for _, r in summary.iterrows():
            out_rows.append({
                "agent_type": agent, "max_steps": max_steps, "mode": r["mode"],
                "alpha": r["alpha"], "n": r["n"], "mean_g": r["g"],
                "success": r["success"], "burden": r["burden"],
                "induced": r["induced"], "backoff": r["backoff"],
            })
        for _, r in band.iterrows():
            out_rows.append({
                "agent_type": agent, "max_steps": max_steps, "mode": r["mode"],
                "alpha": r["alpha"], "bimodality_g": r["bimodality_g"], "sd_burden": r["sd_burden"],
            })
        print()

    pd.DataFrame(out_rows).to_csv("cost_aware_analysis.csv", index=False)
    print("Wrote: cost_aware_analysis.csv")


if __name__ == "__main__":
    main()
