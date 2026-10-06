"""
check_transition_count.py

Tests whether the apparent dwell-length weakening trend is actually a
transition-count effect: at 16-32 step episodes, larger L doesn't just
mean coarser blocks, it mechanically leaves room for far fewer RF<->DF
transitions per episode. Paper 3's own mechanism section frames the
switching-relief effect as coming from REPEATED RE-ENTRY into RF, not
occupancy alone -- transition count is the more precise variable to
check, and it's directly derivable from already-collected columns
(rf_steps + df_steps = realized steps taken; dividing by dwell_length
approximates how many dwell blocks -- and therefore transitions -- an
episode actually completed before ending).

No rerun needed -- this reads the same switching_runs.csv already on disk.

Usage:
    python3 check_transition_count.py switching_runs.csv
"""

import sys
import numpy as np
import pandas as pd


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 check_transition_count.py switching_runs.csv")
        sys.exit(1)

    df = pd.read_csv(sys.argv[1])
    df["realized_steps"] = df["rf_steps"] + df["df_steps"]
    df["approx_transitions"] = (df["realized_steps"] / df["dwell_length"]).clip(lower=1)

    print("=" * 100)
    print("Mean realized transitions per episode, by dwell_length (collapsing across agent_type/max_steps)")
    print("=" * 100)
    summary = df.groupby("dwell_length").agg(
        mean_realized_steps=("realized_steps", "mean"),
        mean_approx_transitions=("approx_transitions", "mean"),
        median_approx_transitions=("approx_transitions", "median"),
        frac_lt_2_transitions=("approx_transitions", lambda x: (x < 2).mean()),
    )
    print(summary.to_string(float_format=lambda x: f"{x:.3f}"))

    print()
    print("=" * 100)
    print("hard_blocks vs approx_transitions, correlation (pooled across dwell_length,")
    print("agent_type, max_steps) -- if transitions is the real driver, this should show")
    print("a clear relationship even INSIDE a fixed dwell_length, not just across them.")
    print("=" * 100)
    for agent_type in df["agent_type"].unique():
        sub = df[df["agent_type"] == agent_type]
        corr = sub[["approx_transitions", "hard_blocks"]].corr().iloc[0, 1]
        print(f"{agent_type}: corr(approx_transitions, hard_blocks) = {corr:.4f} (n={len(sub)})")

    print()
    print("=" * 100)
    print("Binned by approx_transitions directly (pooled across dwell_length) --")
    print("this is the real test: does hard_blocks order by TRANSITIONS regardless")
    print("of which dwell_length produced that transition count?")
    print("=" * 100)
    df["transition_bin"] = pd.cut(df["approx_transitions"], bins=[0, 1, 2, 3, 5, 8, 100],
                                    labels=["1", "2", "3", "4-5", "6-8", "9+"])
    binned = df.groupby(["agent_type", "transition_bin"], observed=True).agg(
        n=("hard_blocks", "size"),
        mean_hard_blocks=("hard_blocks", "mean"),
        mean_dwell_length=("dwell_length", "mean"),
    )
    print(binned.to_string(float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
