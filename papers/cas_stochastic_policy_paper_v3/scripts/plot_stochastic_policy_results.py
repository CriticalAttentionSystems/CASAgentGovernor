#!/usr/bin/env python3

import csv
from pathlib import Path

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT
FIGURE_DIR = OUTPUT_DIR / "figures"
AGGREGATE_CSV = OUTPUT_DIR / "stochastic_policy_aggregate.csv"

AGENT_ORDER = ["exploratory", "clinging", "persistent", "impulsive", "noisy"]
AGENT_LABELS = {
    "exploratory": "Exploratory",
    "clinging": "Clinging",
    "persistent": "Persistent",
    "impulsive": "Impulsive",
    "noisy": "Noisy",
}


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def f(row, key):
    return float(row[key])


def rows_by_agent_condition(rows):
    return {(row["agent_type"], row["condition"]): row for row in rows}


def ordered_agents(rows):
    present = {row["agent_type"] for row in rows}
    return [agent for agent in AGENT_ORDER if agent in present]


def grouped_bar(rows, metric_key, sem_key, ylabel, title, filename, ylim_pad=1.20, value_format="{:.2f}"):
    by_key = rows_by_agent_condition(rows)
    agents = ordered_agents(rows)
    x = list(range(len(agents)))
    width = 0.38

    ungoverned = [f(by_key[(a, "ungoverned")], metric_key) for a in agents]
    governed = [f(by_key[(a, "governed")], metric_key) for a in agents]
    ungoverned_sem = [f(by_key[(a, "ungoverned")], sem_key) if sem_key else 0.0 for a in agents]
    governed_sem = [f(by_key[(a, "governed")], sem_key) if sem_key else 0.0 for a in agents]
    labels = [AGENT_LABELS.get(a, a) for a in agents]

    fig, ax = plt.subplots(figsize=(9, 5.4))
    ax.bar([i - width / 2 for i in x], ungoverned, width, yerr=ungoverned_sem if sem_key else None,
           capsize=4 if sem_key else 0, label="Ungoverned")
    ax.bar([i + width / 2 for i in x], governed, width, yerr=governed_sem if sem_key else None,
           capsize=4 if sem_key else 0, label="Governed")

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ymax = max(max(ungoverned), max(governed))
    ax.set_ylim(0, ymax * ylim_pad if ymax > 0 else 1)

    for i, value in enumerate(ungoverned):
        ax.text(i - width / 2, value, value_format.format(value), ha="center", va="bottom", fontsize=8)
    for i, value in enumerate(governed):
        ax.text(i + width / 2, value, value_format.format(value), ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    out = FIGURE_DIR / filename
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Wrote {out}")


def success_rate_plot(rows):
    grouped_bar(rows, "success_rate", None, "Success rate", "Success rate by agent failure mode",
                "success_rate_by_agent_type.png", ylim_pad=1.08, value_format="{:.2f}")


def burden_plot(rows):
    grouped_bar(rows, "mean_post_error_tool_burden", "sem_post_error_tool_burden",
                "Mean post-error tool burden", "CAS regulation reduces post-error burden",
                "post_error_burden_by_agent_type.png")


def failed_calls_plot(rows):
    grouped_bar(rows, "mean_failed_tool_calls", "sem_failed_tool_calls", "Mean failed tool calls",
                "CAS regulation reduces failed tool calls", "failed_tool_calls_by_agent_type.png")


def repeated_failed_plot(rows):
    by_key = rows_by_agent_condition(rows)
    agents = ordered_agents(rows)
    x = list(range(len(agents)))
    width = 0.38
    visible_zero_height = 0.035

    ungoverned = [f(by_key[(a, "ungoverned")], "mean_repeated_failed_actions") for a in agents]
    governed = [f(by_key[(a, "governed")], "mean_repeated_failed_actions") for a in agents]
    ungoverned_sem = [f(by_key[(a, "ungoverned")], "sem_repeated_failed_actions") for a in agents]
    governed_sem = [f(by_key[(a, "governed")], "sem_repeated_failed_actions") for a in agents]
    labels = [AGENT_LABELS.get(a, a) for a in agents]
    governed_display = [visible_zero_height if abs(value) < 1.0e-12 else value for value in governed]

    fig, ax = plt.subplots(figsize=(9, 5.4))
    ax.bar([i - width / 2 for i in x], ungoverned, width, yerr=ungoverned_sem,
           capsize=4, label="Ungoverned")
    ax.bar([i + width / 2 for i in x], governed_display, width, yerr=governed_sem,
           capsize=4, label="Governed")

    for i, value in enumerate(ungoverned):
        ax.text(i - width / 2, value, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    for i, value in enumerate(governed):
        display_value = governed_display[i]
        ax.text(i + width / 2, display_value + 0.035, f"{value:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("Mean repeated failed actions")
    ax.set_title("CAS regulation suppresses repeated failed actions")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ymax = max(max(ungoverned), max(governed_display))
    ax.set_ylim(0, ymax * 1.20 if ymax > 0 else 1)

    fig.tight_layout()
    out = FIGURE_DIR / "repeated_failed_actions_by_agent_type.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Wrote {out}")


def total_tool_calls_plot(rows):
    grouped_bar(rows, "mean_tool_calls", "sem_tool_calls", "Mean total tool calls",
                "Total tool calls by agent failure mode", "total_tool_calls_by_agent_type.png")


def burden_reduction_plot(rows):
    by_key = rows_by_agent_condition(rows)
    agents = ordered_agents(rows)
    reductions = []
    for agent in agents:
        ungoverned = f(by_key[(agent, "ungoverned")], "mean_post_error_tool_burden")
        governed = f(by_key[(agent, "governed")], "mean_post_error_tool_burden")
        reductions.append(100.0 * (ungoverned - governed) / ungoverned if ungoverned > 0 else 0.0)

    labels = [AGENT_LABELS.get(a, a) for a in agents]
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.bar(labels, reductions)
    ax.axhline(0.0, linewidth=1)
    ax.set_ylabel("Post-error burden reduction (%)")
    ax.set_title("CAS burden reduction depends on agent failure mode")
    ymin = min(reductions)
    ymax = max(reductions)
    ax.set_ylim(min(0, ymin) * 1.20 - 5, max(0, ymax) * 1.20 + 5)
    for i, value in enumerate(reductions):
        va = "bottom" if value >= 0 else "top"
        offset = 1.0 if value >= 0 else -1.0
        ax.text(i, value + offset, f"{value:.1f}%", ha="center", va=va)

    fig.tight_layout()
    out = FIGURE_DIR / "post_error_burden_reduction_percent.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Wrote {out}")


def success_change_plot(rows):
    by_key = rows_by_agent_condition(rows)
    agents = ordered_agents(rows)
    changes = []
    for agent in agents:
        ungoverned = f(by_key[(agent, "ungoverned")], "success_rate")
        governed = f(by_key[(agent, "governed")], "success_rate")
        changes.append(100.0 * (governed - ungoverned))

    labels = [AGENT_LABELS.get(a, a) for a in agents]
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.bar(labels, changes)
    ax.axhline(0.0, linewidth=1)
    ax.set_ylabel("Success-rate change (percentage points)")
    ax.set_title("CAS success effects depend on agent failure mode")
    ymin = min(changes)
    ymax = max(changes)
    ax.set_ylim(min(0, ymin) * 1.25 - 1, max(0, ymax) * 1.25 + 1)
    for i, value in enumerate(changes):
        va = "bottom" if value >= 0 else "top"
        offset = 0.25 if value >= 0 else -0.25
        ax.text(i, value + offset, f"{value:+.1f}", ha="center", va=va)

    fig.tight_layout()
    out = FIGURE_DIR / "success_rate_change_percentage_points.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Wrote {out}")


def success_vs_burden_tradeoff_plot(rows):
    """Clean Figure 5: success change versus burden reduction.

    Positive x means lower post-error burden under governance. Positive y means
    a higher success rate under governance. This is clearer than the crowded
    start-to-end arrow plot.
    """
    by_key = rows_by_agent_condition(rows)
    agents = ordered_agents(rows)

    points = []
    for agent in agents:
        u = by_key[(agent, "ungoverned")]
        g = by_key[(agent, "governed")]
        burden_u = f(u, "mean_post_error_tool_burden")
        burden_g = f(g, "mean_post_error_tool_burden")
        success_u = f(u, "success_rate")
        success_g = f(g, "success_rate")
        burden_reduction = 100.0 * (burden_u - burden_g) / burden_u if burden_u > 0 else 0.0
        success_change = 100.0 * (success_g - success_u)
        points.append((agent, burden_reduction, success_change))

    fig, ax = plt.subplots(figsize=(8.2, 5.8))
    ax.axhline(0.0, linewidth=1)
    ax.axvline(0.0, linewidth=1)

    label_offsets = {
        "exploratory": (1.3, 0.8),
        "clinging": (1.4, 0.7),
        "persistent": (-12.0, 0.7),
        "impulsive": (1.3, -0.6),
        "noisy": (1.2, 0.7),
    }

    for agent, x, y in points:
        label = AGENT_LABELS.get(agent, agent)
        ax.scatter([x], [y], s=90)
        dx, dy = label_offsets.get(agent, (1.0, 0.6))
        ax.text(
            x + dx,
            y + dy,
            f"{label}\n{x:.1f}% burden, {y:+.1f} pp success",
            va="center",
            fontsize=8.5,
        )

    ax.set_xlabel("Post-error burden reduction under governance (%)")
    ax.set_ylabel("Success-rate change under governance (percentage points)")
    ax.set_title("Success and burden changes by agent failure mode")
    ax.set_xlim(-22, 82)
    ax.set_ylim(-5, 26)

    fig.tight_layout()
    out = FIGURE_DIR / "success_vs_post_error_burden_tradeoff.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Wrote {out}")


def write_summary_table(rows):
    by_key = rows_by_agent_condition(rows)
    agents = ordered_agents(rows)
    out = FIGURE_DIR / "stochastic_summary_table.md"
    lines = []
    lines.append("| Agent type | Success ungoverned | Success governed | Burden ungoverned | Burden governed | Burden reduction |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for agent in agents:
        u = by_key[(agent, "ungoverned")]
        g = by_key[(agent, "governed")]
        success_u = f(u, "success_rate")
        success_g = f(g, "success_rate")
        burden_u = f(u, "mean_post_error_tool_burden")
        burden_g = f(g, "mean_post_error_tool_burden")
        reduction = 100.0 * (burden_u - burden_g) / burden_u if burden_u > 0 else 0.0
        lines.append(
            f"| {AGENT_LABELS.get(agent, agent)} | {success_u:.3f} | {success_g:.3f} | "
            f"{burden_u:.3f} | {burden_g:.3f} | {reduction:.1f}% |"
        )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out}")


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    if not AGGREGATE_CSV.exists():
        raise FileNotFoundError(f"Could not find aggregate CSV: {AGGREGATE_CSV}")
    rows = read_csv(AGGREGATE_CSV)
    success_rate_plot(rows)
    burden_plot(rows)
    failed_calls_plot(rows)
    repeated_failed_plot(rows)
    total_tool_calls_plot(rows)
    burden_reduction_plot(rows)
    success_change_plot(rows)
    success_vs_burden_tradeoff_plot(rows)
    write_summary_table(rows)


if __name__ == "__main__":
    main()
