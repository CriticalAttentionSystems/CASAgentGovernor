Recorded numerical results accompanying
When the Governor Becomes the Disturbance

gemini_induced_checkpoint.csv
  Direct Gemini 2.5 Flash experiment: 576 episode records from six tasks,
  four governor modes, three induced-cost scales, and eight repeats.
  All six tasks contribute to Table 3 and are displayed separately in Table 4.
  At alpha=2, C003 mean hard blocks are 5.375 (blind) and 0.250 (aware_k3.0).
  Its answered counts are 1/8 and 8/8; substring-scored successes remain 0/8
  in both conditions. Answer correctness cannot be inferred from answering alone.
  These are recorded episode metrics, not model response transcripts.

induced_robustness_mechanisms.csv
  Stochastic aggregate results underlying Table 1.

induced_robustness_baselines.csv
  Stochastic aggregate results underlying Table 2.

cost_aware_aggregate.csv
  Stochastic backoff sweep. Section 5.2 uses RF, 16-step entries.

Regenerate numerical summaries with Python 3, using no additional packages:
  python3 anc/summarize_results.py

The CSV files are unchanged copies of the recorded experiment outputs.
No new experiments or reviewer diagnostic results are included here.
