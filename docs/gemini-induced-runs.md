# Safe induced-cost runs

The archived `output/gemini_induced_checkpoint.csv` and the manuscript's `anc/`
data are historical results. They are not modified or automatically adopted by
the new harness: they lack the provenance and lifetime API-attempt ledger needed
for safe resumption. This safeguard update does not change the published data.

## Offline checks

Requires the project's Java 25 and Maven dependencies (including IRAM-Omega-Q).

```sh
mvn test
mvn exec:java -Dexec.mainClass=org.cas.agentgovernor.experiment.RunGeminiInducedCost -Dexec.args="mock --run-id smoke"
```

Mock runs make no API calls. Results are stored in
`output/gemini_induced_runs/mock/smoke/checkpoint.csv`. Repeat the same command
to resume/skip completed episodes. Live and mock runs never share checkpoints
or workspaces, even if their run IDs match.

## Live runs (incur API usage)

Only start a new run intentionally, with credentials configured outside source
control. For example:

```sh
mvn exec:java -Dexec.mainClass=org.cas.agentgovernor.experiment.RunGeminiInducedCost -Dexec.args="gemini --run-id confirmation-v2 --max-api-calls 6000"
```

Results are in `output/gemini_induced_runs/gemini/confirmation-v2/checkpoint.csv`.
The run directory also contains `run.properties` (configuration, SHA-256
fingerprint, reserved attempt count, and limit) and a lifetime process lock.
The fingerprint covers settings, prompts, target file contents, scoring answers,
and the relevant compiled implementation classes, including the Gemini prompt
builder. Changed fingerprints refuse resumption rather than mixing experiments.
Recompiling with a different compiler may also change the bytecode fingerprint.
Generation settings remain provider defaults; only the induced/backoff RNG is
seeded. The model alias is not a guarantee of a frozen provider model version.

Resume with the **same backend and run ID**. Omit `--max-api-calls` to retain
the saved ceiling. The count does not reset on restart, provider quota reset,
or an interrupted episode. Completed episodes are skipped; an incomplete episode
restarts from its beginning and its previous attempts remain counted.

Every SDK request attempt, including each outer-loop retry after 429/5xx errors,
must reserve budget on disk before sending. SDK-level retries are disabled.
Failed requests count conservatively whether or not the provider bills them.
A crash between reservation and transmission can overcount by one; the harness
never refunds an uncertain request. This is a **per-run request-attempt limit**,
not a dollar/token limit or an account-wide cap across independent run IDs.

When exhausted, rerunning does not reset the budget. To intentionally permit
additional requests, resume the same run with an explicitly larger
`--max-api-calls` value (a new total ceiling, not an increment).

Do not delete/edit the ledger, copy unrelated CSV rows into a run, or use a new
run ID merely to bypass a limit. Missing metadata, malformed/duplicate rows,
out-of-grid episode IDs, and concurrent writers are rejected. A partial CSV row
is left in place and reported rather than silently marked complete.

## Summarize results

```sh
.venv/bin/python scripts/analyze_gemini_induced.py output/gemini_induced_runs/gemini/confirmation-v2/checkpoint.csv
python3 papers/control_generated_disturbance_arxiv_source/anc/summarize_results.py
```

The second command reproduces the recorded manuscript summaries, including all
six tasks in Table 4. It does not launch experiments.
