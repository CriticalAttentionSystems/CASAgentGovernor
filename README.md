## Results: Controlled Mock-Agent Recovery Experiment

The first experiment evaluates whether the CAS Agent Governor reduces recovery burden in a tool-using LLM-style agent after an initial failed action.

The task setting is deliberately controlled. Each task asks the agent to locate and summarize a file in a small workspace. The user prompt contains an incorrect or incomplete file assumption, such as a wrong filename, wrong folder, stale filename, wrong extension, or nested path. The agent must recover by inspecting the workspace and reading the correct target file.

Two mock agent regimes are tested:

- **Clinging mock:** repeats the first failed action once before recovery.
- **Persistent mock:** repeats the first failed action several times before recovery.

Each regime is run in two conditions:

- **Ungoverned:** the agent follows its default action pattern.
- **Governed:** the CAS Governor evaluates the trace, emits regulatory state signals, and modulates the agent away from blind retries.

The governor uses the state sequence:

```text
NORMAL → DISTURBED → RECOVERING → RELEASED
