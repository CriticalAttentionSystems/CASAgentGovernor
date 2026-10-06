#!/usr/bin/env python3

import csv
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "output"
FIGURE_DIR = OUTPUT_DIR / "figures"

AGGREGATE_CSV = OUTPUT_DIR / "mock_task_aggregate_summary.csv"
TASK_CSV = OUTPUT_DIR / "mock_task_comparison_summary.csv"


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def to_float(row, key):
    return float(row[key])


def condition_label(row):
    mock = row["mock_type"].capitalize()
    agent = row["agent"].replace("_mock", "").capitalize()
    return f"{mock}\n{agent}"


def plot_aggregate_metric(rows, metric_key, ylabel, title, filename):
    labels = [condition_label(row) for row in rows]
    values = [to_float(row, metric_key) for row in rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, values)

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, max(values) * 1.25 if values else 1)

    for i, value in enumerate(values):
        ax.text(i, value, f"{value:.2f}", ha="center", va="bottom")

    fig.tight_layout()
    out = FIGURE_DIR / filename
    fig.savefig(out, dpi=300)
    plt.close(fig)

    print(f"Wrote {out}")


def short_task_label(task_id):
    labels = {
        "T001_wrong_filename": "T001\nWrong filename",
        "T002_wrong_folder": "T002\nWrong folder",
        "T003_wrong_extension": "T003\nWrong extension",
        "T004_empty_expected_directory": "T004\nEmpty directory",
        "T005_similar_competing_filenames": "T005\nSimilar names",
        "T006_old_filename_mentioned": "T006\nOld filename",
        "T007_nested_directory": "T007\nNested directory",
        "T008_latest_summary": "T008\nLatest summary",
    }
    return labels.get(task_id, task_id)


def plot_task_level_burden(task_rows, mock_type, filename):
    selected = [row for row in task_rows if row["mock_type"] == mock_type]

    by_task = defaultdict(dict)

    for row in selected:
        task_id = row["task_id"]
        agent = row["agent"].replace("_mock", "")
        burden = float(row["post_error_tool_burden"])
        by_task[task_id][agent] = burden

    task_ids = list(by_task.keys())
    labels = [short_task_label(t) for t in task_ids]

    y = list(range(len(task_ids)))

    ungoverned = [by_task[t].get("ungoverned", 0.0) for t in task_ids]
    governed = [by_task[t].get("governed", 0.0) for t in task_ids]

    height = 0.38

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh([i + height / 2 for i in y], ungoverned, height, label="Ungoverned")
    ax.barh([i - height / 2 for i in y], governed, height, label="Governed")

    ax.set_xlabel("Post-error tool burden")
    ax.set_ylabel("")
    ax.set_title("CAS governor reduces post-error burden across persistent-agent tasks")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.legend()

    for i, value in enumerate(ungoverned):
        ax.text(value + 0.05, i + height / 2, f"{value:.0f}", va="center")

    for i, value in enumerate(governed):
        ax.text(value + 0.05, i - height / 2, f"{value:.0f}", va="center")

    fig.tight_layout()
    out = FIGURE_DIR / filename
    fig.savefig(out, dpi=300)
    plt.close(fig)

    print(f"Wrote {out}")


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    aggregate_rows = read_csv(AGGREGATE_CSV)
    task_rows = read_csv(TASK_CSV)

    plot_aggregate_metric(
        aggregate_rows,
        "mean_post_error_tool_burden",
        "Mean post-error tool burden",
        "CAS governor reduces post-error tool burden",
        "mean_post_error_burden.png",
    )

    plot_aggregate_metric(
        aggregate_rows,
        "mean_failed_tool_calls",
        "Mean failed tool calls",
        "CAS governor reduces failed tool calls",
        "mean_failed_tool_calls.png",
    )

    plot_aggregate_metric(
        aggregate_rows,
        "mean_repeated_failed_actions",
        "Mean repeated failed actions",
        "CAS governor eliminates repeated failed actions",
        "mean_repeated_failed_actions.png",
    )

    plot_aggregate_metric(
        aggregate_rows,
        "mean_tool_calls",
        "Mean total tool calls",
        "CAS governor reduces total tool calls",
        "mean_total_tool_calls.png",
    )

    plot_task_level_burden(
        task_rows,
        "persistent",
        "persistent_task_level_burden.png",
    )


if __name__ == "__main__":
    main()