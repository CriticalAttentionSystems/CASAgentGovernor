"""
analyze_gemini_induced.py

Analysis for the real-LLM (Gemini) induced-cost pilot
(RunGeminiInducedCost). Small-N by design -- report effect direction and
per-condition means, not significance stars.

Reports, by governor mode (blind vs cost-aware) and induced-cost scale alpha:
  - success rate and answered rate
  - mean induced failures (control-generated disturbance actually injected)
  - mean regulation intensity g (paper-4 suppression = falls with alpha under
    the cost-aware governor, stays pinned under blind)
  - mean backoff steps (cost-aware only)
  - mean post-error tool burden and hard blocks

The two questions this pilot puts to a REAL LLM agent:
  1. Amplification: does raising alpha (the governor's own tightening
     corrupting actions) collapse the LLM's task success under the cost-blind
     governor?
  2. Rescue: does a cost-aware governor that backs off recover that success?

Usage:
    python3 analyze_gemini_induced.py output/gemini_induced_checkpoint.csv
"""

import sys
import pandas as pd


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 analyze_gemini_induced.py gemini_induced_checkpoint.csv")
        sys.exit(1)

    d = pd.read_csv(sys.argv[1])
    d["success"] = d["success"].astype(str).str.lower().eq("true")
    d["answered"] = d["answered"].astype(str).str.lower().eq("true")

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)

    summary = d.groupby(["mode", "alpha"]).agg(
        n=("repeat", "size"),
        success=("success", "mean"),
        answered=("answered", "mean"),
        induced=("induced_failures", "mean"),
        g=("mean_regulation_intensity", "mean"),
        backoff=("backoff_steps", "mean"),
        burden=("post_error_tool_burden", "mean"),
        hard_blocks=("hard_blocks", "mean"),
    ).reset_index()

    print("=" * 100)
    print("Gemini induced-cost pilot: real-LLM control-generated-disturbance test")
    print("=" * 100)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\n" + "=" * 100)
    print("[1] Amplification (blind): success vs alpha  |  [2] Rescue: aware success vs blind")
    print("=" * 100)
    piv = summary.pivot(index="alpha", columns="mode", values="success")
    print("success rate by alpha x mode:")
    print(piv.to_string(float_format=lambda x: f"{x:.3f}"))

    if "blind" in piv.columns:
        blind = piv["blind"]
        drop = blind.iloc[0] - blind.iloc[-1]
        print(f"\n  blind success drop across alpha: {drop:+.3f} "
              f"({'amplification/collapse present' if drop > 0.1 else 'no clear collapse'})")
    # k-sweep: report rescue for every cost-aware mode, best k first.
    aware_cols = sorted([c for c in piv.columns if c.startswith("aware")])
    best = None
    for c in aware_cols:
        if "blind" in piv.columns:
            rescue = (piv[c] - piv["blind"]).mean()
            verdict = "rescue" if rescue > 0.05 else "no rescue" if rescue < -0.05 else "neutral"
            print(f"  mean success rescue ({c} - blind) across alpha: {rescue:+.3f} ({verdict})")
            if best is None or rescue > best[1]:
                best = (c, rescue)
    if best is not None:
        print(f"  --> best backoff sensitivity: {best[0]} (mean rescue {best[1]:+.3f})")

    print("\n" + "=" * 100)
    print("[3] Suppression: regulation intensity g vs alpha (paper-4 = falls under cost-aware)")
    print("=" * 100)
    pg = summary.pivot(index="alpha", columns="mode", values="g")
    print(pg.to_string(float_format=lambda x: f"{x:.3f}"))
    for c in pg.columns:
        gdrop = pg[c].max() - pg[c].iloc[-1]
        print(f"  {c}: g drop across alpha = {gdrop:+.3f} "
              f"({'suppression' if gdrop > 0.03 else 'pinned (no suppression)'})")


if __name__ == "__main__":
    main()
