# 1. Regenerate a complete review from your logs with the bugfixed script
#    (into a NEW file so it doesn't clobber your existing graded work):
python grade_gemini_answers.py logs/gemini_run_*.log --csv review_fresh.csv

# 2. Find just the new rows (should be exactly the 3 T001 rows I identified,
#    assuming no other episodes have run since):
python diff_and_merge_review.py diff \
    --existing review_graded_so_far.csv \
    --fresh review_fresh.csv \
    --out review_diff.csv

# 3. Grade only what's in review_diff.csv (3 rows, or paste them here and
#    I'll do it the same way I did the rest)

# 4. Merge back into a complete graded file:
python diff_and_merge_review.py merge \
    --existing review_graded_so_far.csv \
    --diff review_diff.csv \
    --out review_complete.csv

# 5. Final numbers:
python aggregate_human_grades.py review_complete.csv
