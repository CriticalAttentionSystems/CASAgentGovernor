#!/usr/bin/env python3
"""
aggregate_human_grades.py

Reads back a review.csv produced by grade_gemini_answers.py, after you've
filled in the "your_judgment" column by hand, and computes the real
success rate per condition from YOUR judgments -- the number this
pipeline should have produced in the first place, before the substring
grader's paraphrase blindness got in the way.

Accepts a few spellings for the your_judgment column so you don't have to
be perfectly consistent while filling it in by hand:
  true-ish:  TRUE, True, true, T, Y, Yes, 1, PASS, CORRECT
  false-ish: FALSE, False, false, F, N, No, 0, FAIL, INCORRECT, WRONG
Blank cells (not yet graded) are reported separately and excluded from the
computed rates, rather than silently counted as failures -- so you can run
this partway through grading and see exactly how much is left to do.

Usage:
    python3 aggregate_human_grades.py review.csv
"""

import argparse
import csv
import sys
from collections import defaultdict

TRUE_VALUES = {"true", "t", "y", "yes", "1", "pass", "correct"}
FALSE_VALUES = {"false", "f", "n", "no", "0", "fail", "incorrect", "wrong"}


def parse_judgment(raw):
    """Returns True, False, or None (not yet graded / unrecognized)."""
    if raw is None:
        return None
    cleaned = raw.strip().lower()
    if cleaned == "":
        return None
    if cleaned in TRUE_VALUES:
        return True
    if cleaned in FALSE_VALUES:
        return False
    return None


def load_reviewed(path):
    rows = []
    unrecognized = []

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "your_judgment" not in reader.fieldnames:
            print(
                f"ERROR: {path} has no 'your_judgment' column. Is this a "
                f"review.csv produced by grade_gemini_answers.py?",
                file=sys.stderr,
            )
            sys.exit(1)

        for row in reader:
            raw = row.get("your_judgment", "")
            judgment = parse_judgment(raw)

            if judgment is None and raw.strip() != "":
                unrecognized.append((row["task_id"], row["condition"], row["repeat"], raw))

            row["_judgment"] = judgment
            rows.append(row)

    return rows, unrecognized


def summarize(rows):
    by_condition = defaultdict(lambda: {"graded": 0, "correct": 0, "ungraded": 0, "total": 0})
    by_task_condition = defaultdict(lambda: {"graded": 0, "correct": 0, "ungraded": 0, "total": 0})

    for row in rows:
        condition = row["condition"]
        task_id = row["task_id"]
        judgment = row["_judgment"]

        by_condition[condition]["total"] += 1
        by_task_condition[(task_id, condition)]["total"] += 1

        if judgment is None:
            by_condition[condition]["ungraded"] += 1
            by_task_condition[(task_id, condition)]["ungraded"] += 1
        else:
            by_condition[condition]["graded"] += 1
            by_task_condition[(task_id, condition)]["graded"] += 1
            if judgment:
                by_condition[condition]["correct"] += 1
                by_task_condition[(task_id, condition)]["correct"] += 1

    return by_condition, by_task_condition


def print_report(rows, unrecognized, by_condition, by_task_condition):
    total = len(rows)
    total_graded = sum(1 for r in rows if r["_judgment"] is not None)
    total_ungraded = total - total_graded

    print("=" * 78)
    print("HUMAN-GRADED SUCCESS RATES (from your_judgment column)")
    print("=" * 78)

    condition_order = {"ungoverned": 0, "governed": 1, "sham_regulation": 2}
    for condition in sorted(by_condition, key=lambda c: condition_order.get(c, 99)):
        stats = by_condition[condition]
        if stats["graded"] == 0:
            print(f"\n{condition}: 0/{stats['total']} graded yet -- nothing to report")
            continue
        rate = stats["correct"] / stats["graded"]
        print(f"\n{condition}:")
        print(f"  human-graded success rate: {stats['correct']}/{stats['graded']} = {rate:.3f}"
              f"  ({stats['ungraded']} of {stats['total']} not yet graded)")

    print("\n" + "-" * 78)
    print("Per task x condition breakdown:")
    print("-" * 78)
    current_task = None
    for (task_id, condition) in sorted(by_task_condition, key=lambda tc: (tc[0], condition_order.get(tc[1], 99))):
        if task_id != current_task:
            current_task = task_id
            print(f"\n{task_id}")
        stats = by_task_condition[(task_id, condition)]
        if stats["graded"] == 0:
            print(f"  {condition:18s} 0/{stats['total']} graded yet")
            continue
        rate = stats["correct"] / stats["graded"]
        print(f"  {condition:18s} {stats['correct']}/{stats['graded']} = {rate:.3f}"
              f"  ({stats['ungraded']} ungraded)")

    print("\n" + "=" * 78)
    print(f"TOTAL: {total_graded}/{total} rows graded ({total_ungraded} remaining).")
    if total_ungraded > 0:
        print(f"Fill in the remaining {total_ungraded} 'your_judgment' cells in your "
              f"review CSV and re-run this script for a complete picture.")
    print("=" * 78)

    if unrecognized:
        print(f"\nWARNING: {len(unrecognized)} row(s) had a your_judgment value that "
              f"wasn't recognized as true/false and were treated as ungraded:")
        for task_id, condition, repeat, raw in unrecognized:
            print(f"  {task_id} / {condition} / repeat {repeat}: your_judgment='{raw}'")
        print(f"Recognized values -- true-ish: {sorted(TRUE_VALUES)} / "
              f"false-ish: {sorted(FALSE_VALUES)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("review_csv", help="Path to your hand-graded review.csv")
    args = parser.parse_args()

    rows, unrecognized = load_reviewed(args.review_csv)
    by_condition, by_task_condition = summarize(rows)
    print_report(rows, unrecognized, by_condition, by_task_condition)


if __name__ == "__main__":
    main()
