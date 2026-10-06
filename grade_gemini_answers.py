#!/usr/bin/env python3
"""
grade_gemini_answers.py

AgentRunMetrics.success() grades a real Gemini answer by checking whether
it contains an exact, hand-written expectedAnswer string. Real LLM answers
paraphrase even when substantively correct (confirmed directly: T007's
"The run succeeded: true" is right but fails the check, since the expected
string is "The run succeeded." with a period, not a colon). That means the
success_rate column in the checkpoint CSV is not trustworthy as a
correctness measure for this real-Gemini arm -- see the conversation this
script came out of for the full diagnosis across all 8 tasks.

This script does NOT try to fix the grader. It parses every completed
episode out of your gemini_run_*.log files and lines up each logged final
answer next to its task's actual expected answer, so a human (you) can do
a fast, reliable pass and judge correctness by eye -- which, for 71
episodes total, is faster and more trustworthy right now than building a
proper LLM-judge or embedding-similarity grader would be.

It also reports the naive substring-match verdict per row (the same logic
AgentRunMetrics.success() uses) so you can see exactly where the automated
grader and your own judgment disagree, which is itself useful evidence for
how bad the grading gap actually is.

Usage:
    # Print a readable side-by-side review to the terminal:
    python3 grade_gemini_answers.py logs/gemini_run_*.log

    # Also write a CSV with a blank "your_judgment" column to fill in by
    # hand (TRUE/FALSE), so you can recompute real success rates afterward:
    python3 grade_gemini_answers.py logs/gemini_run_*.log --csv review.csv

    # Only show rows where the naive grader and you might disagree, i.e.
    # everything NOT already an exact substring match (usually the
    # interesting ones):
    python3 grade_gemini_answers.py logs/gemini_run_*.log --only-mismatches
"""

import argparse
import csv
import glob
import re
import sys

# Pulled directly from src/main/java/org/cas/agentgovernor/experiment/tasks/MockTaskSet.java.
# If you add/change tasks, update this dict to match -- this script has no
# way to read the Java source itself.
EXPECTED_ANSWERS = {
    "T001_wrong_filename": (
        "The trial config defines three phases: baseline from 0 to 120 "
        "seconds, disturbance from 120 to 240 seconds, and recovery from "
        "240 to 420 seconds."
    ),
    "T002_wrong_folder": "The mean post-error burden is 3.7.",
    "T003_wrong_extension": "The condition is breath_recovery.",
    "T004_empty_expected_directory": "The run succeeded.",
    "T005_similar_competing_filenames": (
        "Session 001 completed successfully with post-error burden 3.1."
    ),
    "T006_old_filename_mentioned": (
        "The current config defines baseline, disturbance, and recovery phases."
    ),
    "T007_nested_directory": "The run succeeded.",
    "T008_latest_summary": "The latest post-error burden is 2.6.",
}

# Matches one episode's block in a gemini_run_*.log file, e.g.:
#
#   ========================================
#   GEMINI | TASK T001_wrong_filename | governed | repeat 0
#   ========================================
#   === GOVERNED FINAL ANSWER ===
#   The trial config file is `configs/trial_config_demo.json`. ...
#   Pausing 15s before next run...
#
# Answers were observed to print as a single physical line (literal "\n"
# sequences inside the text, not real newlines), so the answer is simply
# "the next non-blank line after the FINAL ANSWER header" -- this does NOT
# anchor on what comes after the answer (e.g. "Pausing"), because the very
# LAST episode in any log file has no "Pausing" line after it (the run
# just ends) and an earlier version of this regex silently dropped that
# episode every time, undercounting every log file by one.
BLOCK_RE = re.compile(
    r"GEMINI \| TASK (?P<task_id>\S+) \| (?P<condition>\S+) \| repeat (?P<repeat>\d+)\s*\n"
    r"={5,}\s*\n"
    r"=== \S+ FINAL ANSWER ===\s*\n"
    r"(?P<answer>[^\n]*)",
)


def parse_logs(paths):
    records = []

    for path in paths:
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
        except OSError as e:
            print(f"WARNING: could not read {path}: {e}", file=sys.stderr)
            continue

        for m in BLOCK_RE.finditer(text):
            task_id = m.group("task_id")
            condition = m.group("condition")
            repeat = int(m.group("repeat"))
            answer = m.group("answer").strip()

            expected = EXPECTED_ANSWERS.get(task_id)
            if expected is None:
                print(
                    f"WARNING: no expected answer on file for task_id="
                    f"'{task_id}' (seen in {path}) -- add it to "
                    f"EXPECTED_ANSWERS in this script.",
                    file=sys.stderr,
                )
                expected = ""

            substring_match = bool(expected) and expected.lower() in answer.lower()
            timed_out = answer.strip().lower().startswith("stopped:")

            records.append({
                "task_id": task_id,
                "condition": condition,
                "repeat": repeat,
                "expected_answer": expected,
                "actual_answer": answer,
                "timed_out": timed_out,
                "substring_match": substring_match,
                "your_judgment": "",  # left blank for manual fill-in
                "source_file": path,
            })

    # Stable, readable ordering: task, then condition, then repeat.
    condition_order = {"ungoverned": 0, "governed": 1, "sham_regulation": 2}
    records.sort(key=lambda r: (
        r["task_id"],
        condition_order.get(r["condition"], 99),
        r["repeat"],
    ))

    return records


def print_review(records, only_mismatches):
    current_task = None

    for r in records:
        if only_mismatches and r["substring_match"]:
            continue

        if r["task_id"] != current_task:
            current_task = r["task_id"]
            print("\n" + "=" * 78)
            print(current_task)
            print("EXPECTED: " + r["expected_answer"])
            print("=" * 78)

        flag = "TIMED OUT" if r["timed_out"] else ("PASS (substring)" if r["substring_match"] else "FAIL (substring)")
        print(f"\n[{r['condition']} / repeat {r['repeat']}]  {flag}")
        print("  " + r["actual_answer"])

    if not records:
        print("No completed episodes found in the given log file(s).")
        return

    total = len(records)
    passed = sum(1 for r in records if r["substring_match"])
    timed_out = sum(1 for r in records if r["timed_out"])
    print("\n" + "-" * 78)
    print(f"{total} episodes parsed. Naive substring grader: {passed}/{total} pass "
          f"({100 * passed / total:.1f}%). {timed_out} timed out without answering.")
    print("This substring rate should roughly match what's already in your "
          "checkpoint CSV -- if it doesn't, something about this script's "
          "parsing or your log files differs from what's expected; treat "
          "the checkpoint CSV as ground truth for that number, and this "
          "script only for the side-by-side text.")
    print("-" * 78)


def write_csv(records, path):
    if not records:
        print("Nothing to write (no episodes parsed).", file=sys.stderr)
        return

    fieldnames = [
        "task_id", "condition", "repeat",
        "expected_answer", "actual_answer",
        "timed_out", "substring_match", "your_judgment",
        "source_file",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"Wrote {len(records)} rows to {path}. Fill in the 'your_judgment' "
          f"column by hand (e.g. TRUE/FALSE) for a real, human-graded "
          f"success rate you can recompute yourself afterward.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("logs", nargs="+", help="Path(s) or glob pattern(s) to gemini_run_*.log files")
    parser.add_argument("--csv", help="Also write a CSV with a blank your_judgment column")
    parser.add_argument("--only-mismatches", action="store_true",
                         help="Only print rows that FAIL the naive substring check "
                              "(usually the interesting ones worth your attention)")
    args = parser.parse_args()

    paths = []
    for pattern in args.logs:
        matched = sorted(glob.glob(pattern))
        if not matched:
            print(f"WARNING: no files matched: {pattern}", file=sys.stderr)
        paths.extend(matched)

    if not paths:
        print("ERROR: no log files found.", file=sys.stderr)
        sys.exit(1)

    records = parse_logs(paths)
    print_review(records, args.only_mismatches)

    if args.csv:
        write_csv(records, args.csv)


if __name__ == "__main__":
    main()
