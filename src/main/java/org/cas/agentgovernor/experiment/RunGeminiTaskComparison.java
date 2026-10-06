/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package org.cas.agentgovernor.experiment;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;

import org.cas.agentgovernor.agents.GovernedLLMAgent;
import org.cas.agentgovernor.agents.UngovernedLLMAgent;
import org.cas.agentgovernor.core.CASGovernor;
import org.cas.agentgovernor.core.RegulatorySignal;
import org.cas.agentgovernor.core.RegulatoryState;
import org.cas.agentgovernor.experiment.tasks.MockTaskSet;
import org.cas.agentgovernor.experiment.tasks.MockTaskSpec;
import org.cas.agentgovernor.llm.GeminiLLMClient;
import org.cas.agentgovernor.llm.GeminiQuotaExhaustedException;
import org.cas.agentgovernor.metrics.AgentRunMetrics;
import org.cas.agentgovernor.tools.FileTool;
import org.cas.agentgovernor.tools.ToolRegistry;

/**
 * Runs the real-Gemini {@code ungoverned} / {@code governed} /
 * {@code sham_regulation} comparison with checkpointing, so free-tier quota
 * exhaustion (RPD in particular -- see GeminiQuotaExhaustedException) does
 * not throw away completed work. Progress is written to
 * {@code output/gemini_task_comparison_checkpoint.csv} after EVERY
 * individual run, and on startup this file is read back to figure out which
 * (task, condition, repeat) combinations are already done.
 *
 * <p><b>sham_regulation (the real-LLM validity check).</b> For each
 * (task, repeat), once {@code governed} completes, its REAL realized
 * per-step regulatory-signal sequence (see
 * {@code GovernedLLMAgent#controlSignalHistory()}) is persisted to a side
 * file under {@code output/gemini_signal_sequences/}. The {@code
 * sham_regulation} run for that same (task, repeat) loads that sequence,
 * shuffles it (seeded deterministically from the task id and repeat), and
 * replays a FRESH real Gemini episode using the shuffled sequence to drive
 * both the prompt's regulatory-state text and the governor's block/warn
 * decisions -- same mechanism, same total intervention frequency, no longer
 * contingent on what actually happened in that specific episode. This is
 * the real-LLM analogue of the synthetic stochastic-policy experiment's
 * sham_regulation condition and answers the same question: is a governed
 * agent's advantage attributable to the signal tracking real disturbance
 * history, or would any similarly-strong, similarly-frequent perturbation
 * do just as well?
 *
 * <p><b>Why the signal sequence is persisted to disk rather than kept only
 * in memory:</b> this pipeline is explicitly designed to be interrupted by
 * daily quota exhaustion and resumed across multiple sessions, possibly
 * days apart (free tier has been observed as low as 20 requests/day). If
 * governed's realized sequence only existed in memory during the run that
 * produced it, an interruption between governed finishing and its paired
 * sham run starting would permanently lose the ability to pair them
 * correctly -- the only fix would be re-spending quota to run governed
 * again. Persisting the sequence the moment governed completes means sham
 * can be run in any later session, using any later day's quota, without
 * ever re-running governed.
 *
 * <p>REPEATS is intentionally small: unlike the synthetic stochastic-policy
 * experiment (thousands of seeds), each Gemini call spends real, scarce
 * free-tier quota. Adding sham_regulation as a third condition increases
 * total planned episodes by 50% (tasks x 3 conditions x REPEATS instead of
 * x2) -- factor that into how many days this will realistically take at
 * your account's actual daily limit. This is still not a statistically
 * powered design on its own -- see prereg_cas_governor.md \u00a75 / Addendum B,
 * which keep the Gemini arm out of confirmatory analysis until enough
 * repeats have accumulated.
 *
 * @author veronique
 */
public class RunGeminiTaskComparison {

    private static final Path WORKSPACE = Path.of("test_workspace");
    private static final Path OUTPUT_DIR = Path.of("output");
    private static final Path CHECKPOINT_CSV = OUTPUT_DIR.resolve("gemini_task_comparison_checkpoint.csv");
    private static final Path SIGNAL_SEQUENCE_DIR = OUTPUT_DIR.resolve("gemini_signal_sequences");

    private static final String MODEL = "gemini-2.5-flash";

    /*
     * Free-tier RPM is typically ~10 for gemini-2.5-flash; this stays well
     * under that. Unlike before, this delay is no longer the only defense
     * against 429s -- GeminiLLMClient itself now retries per-minute 429s
     * with exponential backoff, so this can be modest rather than extremely
     * conservative.
     */
    private static final long DELAY_MILLIS = 8_000L;
    private static final long PAUSE_BETWEEN_RUNS_MILLIS = 15_000L;

    private static final int MAX_STEPS = 16;

    /** Repeats per (task, condition). Kept small on purpose -- see class doc. */
    private static final int REPEATS = 3;

    public static void main(String[] args) throws Exception {
        Files.createDirectories(OUTPUT_DIR);
        Files.createDirectories(SIGNAL_SEQUENCE_DIR);

        Set<String> completed = loadCompletedRunKeys();
        System.out.println("Resuming with " + completed.size() + " runs already completed "
                + "(from " + CHECKPOINT_CSV.toAbsolutePath() + ").");

        ensureCheckpointHeader();

        List<MockTaskSpec> tasks = MockTaskSet.fileRecoveryTasks();

        // ONE client for the entire run, reused across every episode, closed
        // once at the end -- see close()'s doc on GeminiLLMClient for why.
        try (GeminiLLMClient llm = new GeminiLLMClient(MODEL)) {
            try {
                for (MockTaskSpec task : tasks) {
                    for (int repeat = 0; repeat < REPEATS; repeat++) {
                        runOneIfNeeded(llm, task, "ungoverned", repeat, completed);
                        runOneIfNeeded(llm, task, "governed", repeat, completed);
                        runShamIfNeeded(llm, task, repeat, completed);
                    }
                }
            } catch (GeminiQuotaExhaustedException e) {
                System.out.println("\n========================================");
                if (e.isDailyQuotaExhausted()) {
                    System.out.println("STOPPING: Gemini daily quota (RPD) exhausted.");
                    System.out.println("Progress so far is safely checkpointed in:");
                    System.out.println("  " + CHECKPOINT_CSV.toAbsolutePath());
                    System.out.println("Quota resets at midnight Pacific time. Re-run this program");
                    System.out.println("tomorrow (or once quota is available) and it will pick up");
                    System.out.println("exactly where it left off -- already-completed runs will be");
                    System.out.println("skipped automatically, and any governed run that finished");
                    System.out.println("this session already has its signal sequence saved for a");
                    System.out.println("later sham_regulation pairing.");
                } else {
                    System.out.println("STOPPING: Gemini per-minute rate limit (RPM/TPM) was hit");
                    System.out.println("repeatedly even after exponential backoff. Progress so far");
                    System.out.println("is checkpointed in " + CHECKPOINT_CSV.toAbsolutePath() + ".");
                    System.out.println("Re-run this program in a few minutes to resume.");
                }
                System.out.println("========================================");
                System.out.println(e.getMessage());

                writeAggregateFromCheckpoint();
                return;
            }
        }

        System.out.println("\nAll runs complete.");
        writeAggregateFromCheckpoint();
    }

    /**
     * Runs a single (task, condition, repeat) if it isn't already in the
     * checkpoint, then appends its result to the checkpoint file
     * immediately -- so if the very next call throws
     * GeminiQuotaExhaustedException, this run's result is not lost. For
     * condition="governed", also persists its realized signal sequence for
     * a later (possibly much later) paired sham_regulation run.
     */
    private static void runOneIfNeeded(
            GeminiLLMClient llm,
            MockTaskSpec task,
            String condition,
            int repeat,
            Set<String> completed
    ) throws Exception {
        String key = runKey(task.taskId(), condition, repeat);

        if (completed.contains(key)) {
            System.out.println("SKIP (already checkpointed): " + key);
            return;
        }

        System.out.println("\n========================================");
        System.out.println("GEMINI | TASK " + task.taskId() + " | " + condition + " | repeat " + repeat);
        System.out.println("========================================");

        resetWorkspace(task);

        AgentRunMetrics metrics;

        if ("governed".equals(condition)) {
            ToolRegistry tools = new ToolRegistry();
            tools.register(new FileTool(WORKSPACE.toString()));
            CASGovernor governor = CASGovernor.forExpectedTarget(task.expectedTarget());

            GovernedLLMAgent agent = new GovernedLLMAgent(llm, governor, tools, DELAY_MILLIS);
            String answer = agent.run(task.prompt(), MAX_STEPS);

            System.out.println("=== GOVERNED FINAL ANSWER ===");
            System.out.println(answer);

            metrics = AgentRunMetrics.fromHistory("governed_gemini", agent.history(), task.expectedAnswer());

            writeSignalSequence(task.taskId(), repeat, agent.controlSignalHistory());
        } else {
            metrics = runUngoverned(llm, task);
        }

        appendCheckpointRow(task.taskId(), condition, repeat, metrics);
        completed.add(key);

        System.out.println("Pausing " + (PAUSE_BETWEEN_RUNS_MILLIS / 1000) + "s before next run...");
        Thread.sleep(PAUSE_BETWEEN_RUNS_MILLIS);
    }

    /**
     * Runs the sham_regulation episode for (task, repeat) if it isn't
     * already checkpointed AND a persisted governed signal sequence exists
     * for that same (task, repeat). If governed hasn't completed yet
     * (e.g. quota ran out before it did, in this or a past session), this
     * is silently skipped for now -- it will be picked up automatically on
     * a future run once governed's sequence file exists.
     */
    private static void runShamIfNeeded(
            GeminiLLMClient llm,
            MockTaskSpec task,
            int repeat,
            Set<String> completed
    ) throws Exception {
        String key = runKey(task.taskId(), "sham_regulation", repeat);

        if (completed.contains(key)) {
            System.out.println("SKIP (already checkpointed): " + key);
            return;
        }

        List<RegulatorySignal> realizedSequence = readSignalSequence(task.taskId(), repeat);

        if (realizedSequence == null) {
            System.out.println("SKIP (governed not yet completed for this task/repeat, "
                    + "no signal sequence to shuffle yet): " + key);
            return;
        }

        if (realizedSequence.isEmpty()) {
            System.out.println("SKIP (governed's signal sequence was empty for): " + key);
            return;
        }

        System.out.println("\n========================================");
        System.out.println("GEMINI | TASK " + task.taskId() + " | sham_regulation | repeat " + repeat);
        System.out.println("========================================");

        resetWorkspace(task);

        List<RegulatorySignal> shuffled = new ArrayList<>(realizedSequence);
        long shuffleSeed = shuffleSeedFor(task.taskId(), repeat);
        Collections.shuffle(shuffled, new Random(shuffleSeed));

        ToolRegistry tools = new ToolRegistry();
        tools.register(new FileTool(WORKSPACE.toString()));
        CASGovernor governor = CASGovernor.forExpectedTarget(task.expectedTarget());

        GovernedLLMAgent agent = new GovernedLLMAgent(llm, governor, tools, DELAY_MILLIS, shuffled);
        String answer = agent.run(task.prompt(), shuffled.size());

        System.out.println("=== SHAM_REGULATION FINAL ANSWER ===");
        System.out.println(answer);

        AgentRunMetrics metrics =
                AgentRunMetrics.fromHistory("sham_regulation_gemini", agent.history(), task.expectedAnswer());

        appendCheckpointRow(task.taskId(), "sham_regulation", repeat, metrics);
        completed.add(key);

        System.out.println("Pausing " + (PAUSE_BETWEEN_RUNS_MILLIS / 1000) + "s before next run...");
        Thread.sleep(PAUSE_BETWEEN_RUNS_MILLIS);
    }

    private static long shuffleSeedFor(String taskId, int repeat) {
        long h = 1125899906842597L;
        h = 31 * h + taskId.hashCode();
        h = 31 * h + repeat;
        h = 31 * h + "sham_shuffle".hashCode();
        return h;
    }

    private static AgentRunMetrics runUngoverned(GeminiLLMClient llm, MockTaskSpec task) throws Exception {
        ToolRegistry tools = new ToolRegistry();
        tools.register(new FileTool(WORKSPACE.toString()));

        UngovernedLLMAgent agent = new UngovernedLLMAgent(
                llm,
                tools,
                DELAY_MILLIS
        );

        String answer = agent.run(task.prompt(), MAX_STEPS);

        System.out.println("=== UNGOVERNED FINAL ANSWER ===");
        System.out.println(answer);

        return AgentRunMetrics.fromHistory("ungoverned_gemini", agent.history(), task.expectedAnswer());
    }

    // ------------------------------------------------------------------
    // Signal sequence persistence (for sham_regulation pairing)
    // ------------------------------------------------------------------

    private static Path signalSequencePath(String taskId, int repeat) {
        String safeName = (taskId + "_repeat" + repeat).replaceAll("[^a-zA-Z0-9_.-]", "_");
        return SIGNAL_SEQUENCE_DIR.resolve(safeName + ".tsv");
    }

    /**
     * One line per step: {@code state<TAB>burden<TAB>instruction}. Instruction
     * text is written last and read with a 3-way limited split, so any commas
     * (CASGovernor's instruction strings contain them, e.g. "inspection,
     * schema/listing, or clarification") or other punctuation in the
     * free-text instruction never need escaping -- only a literal TAB
     * character would break this, and none of CASGovernor's instruction
     * strings contain one.
     */
    private static void writeSignalSequence(String taskId, int repeat, List<RegulatorySignal> sequence) throws IOException {
        List<String> lines = new ArrayList<>();
        for (RegulatorySignal signal : sequence) {
            lines.add(String.join("\t",
                    signal.state().name(),
                    Integer.toString(signal.burden()),
                    signal.instruction()
            ));
        }

        Files.writeString(
                signalSequencePath(taskId, repeat),
                String.join(System.lineSeparator(), lines) + System.lineSeparator(),
                StandardCharsets.UTF_8
        );
    }

    /** Returns null if no sequence file exists yet for this (taskId, repeat). */
    private static List<RegulatorySignal> readSignalSequence(String taskId, int repeat) throws IOException {
        Path path = signalSequencePath(taskId, repeat);

        if (!Files.exists(path)) {
            return null;
        }

        List<RegulatorySignal> sequence = new ArrayList<>();

        for (String line : Files.readAllLines(path, StandardCharsets.UTF_8)) {
            if (line.isBlank()) {
                continue;
            }
            String[] parts = line.split("\t", 3);
            RegulatoryState state = RegulatoryState.valueOf(parts[0]);
            int burden = Integer.parseInt(parts[1]);
            String instruction = parts.length > 2 ? parts[2] : "";
            sequence.add(new RegulatorySignal(state, burden, instruction));
        }

        return sequence;
    }

    // ------------------------------------------------------------------
    // Checkpointing
    // ------------------------------------------------------------------

    private static String runKey(String taskId, String condition, int repeat) {
        return taskId + "|" + condition + "|" + repeat;
    }

    private static String checkpointHeader() {
        return "task_id,condition,repeat," + AgentRunMetrics.emptyHeaderWithoutAgentPrefix();
    }

    private static void ensureCheckpointHeader() throws IOException {
        if (Files.exists(CHECKPOINT_CSV)) {
            return;
        }
        Files.writeString(
                CHECKPOINT_CSV,
                checkpointHeader() + System.lineSeparator(),
                StandardCharsets.UTF_8
        );
    }

    /**
     * Reads the checkpoint CSV (if any) and returns the set of
     * (task, condition, repeat) keys already completed, so they can be
     * skipped on resume. Tolerant of a missing file (first run) or a
     * header-only file.
     */
    private static Set<String> loadCompletedRunKeys() throws IOException {
        Set<String> keys = new HashSet<>();

        if (!Files.exists(CHECKPOINT_CSV)) {
            return keys;
        }

        List<String> lines = Files.readAllLines(CHECKPOINT_CSV, StandardCharsets.UTF_8);

        for (int i = 1; i < lines.size(); i++) { // skip header
            String line = lines.get(i);
            if (line.isBlank()) {
                continue;
            }
            String[] parts = line.split(",", 4);
            if (parts.length < 3) {
                continue;
            }
            keys.add(runKey(parts[0], parts[1], Integer.parseInt(parts[2])));
        }

        return keys;
    }

    /** Appends one completed run's row to the checkpoint file immediately. */
    private static void appendCheckpointRow(
            String taskId,
            String condition,
            int repeat,
            AgentRunMetrics metrics
    ) throws IOException {
        String row = String.join(",",
                taskId,
                condition,
                Integer.toString(repeat),
                metrics.toCsvRow()
        ) + System.lineSeparator();

        Files.writeString(
                CHECKPOINT_CSV,
                row,
                StandardCharsets.UTF_8,
                StandardOpenOption.CREATE,
                StandardOpenOption.APPEND
        );
    }

    // ------------------------------------------------------------------
    // Aggregation (re-derived from the checkpoint file, so it reflects
    // however much progress exists whenever this is run -- complete or not)
    // ------------------------------------------------------------------

    private static void writeAggregateFromCheckpoint() throws IOException {
        if (!Files.exists(CHECKPOINT_CSV)) {
            return;
        }

        List<String> lines = Files.readAllLines(CHECKPOINT_CSV, StandardCharsets.UTF_8);
        if (lines.size() <= 1) {
            System.out.println("No completed runs yet; nothing to aggregate.");
            return;
        }

        List<CheckpointRow> rows = new ArrayList<>();
        String[] header = lines.get(0).split(",");
        int successIdx = indexOf(header, "success");
        int answeredIdx = indexOf(header, "answered");

        for (int i = 1; i < lines.size(); i++) {
            if (lines.get(i).isBlank()) {
                continue;
            }
            String[] parts = lines.get(i).split(",");
            String taskId = parts[0];
            String condition = parts[1];
            boolean success = "true".equalsIgnoreCase(parts[successIdx]);
            boolean answered = "true".equalsIgnoreCase(parts[answeredIdx]);
            rows.add(new CheckpointRow(taskId, condition, success, answered));
        }

        List<String> aggregateRows = new ArrayList<>();
        aggregateRows.add("backend,condition,n,success_rate,answered_rate");

        for (String condition : List.of("ungoverned", "governed", "sham_regulation")) {
            List<CheckpointRow> selected = rows.stream()
                    .filter(r -> r.condition.equals(condition))
                    .toList();

            if (selected.isEmpty()) {
                continue;
            }

            int n = selected.size();
            long successes = selected.stream().filter(r -> r.success).count();
            long answered = selected.stream().filter(r -> r.answered).count();

            aggregateRows.add(String.join(",",
                    "gemini",
                    condition,
                    Integer.toString(n),
                    String.format("%.4f", ((double) successes) / n),
                    String.format("%.4f", ((double) answered) / n)
            ));
        }

        Path aggregateCsv = OUTPUT_DIR.resolve("gemini_task_aggregate_summary.csv");
        Files.writeString(
                aggregateCsv,
                String.join(System.lineSeparator(), aggregateRows) + System.lineSeparator(),
                StandardCharsets.UTF_8
        );

        System.out.println("\n========================================");
        System.out.println("GEMINI AGGREGATE SUMMARY (from checkpoint, may be partial)");
        System.out.println("========================================");
        aggregateRows.forEach(System.out::println);
        System.out.println("\nWrote: " + aggregateCsv.toAbsolutePath());
    }

    private static int indexOf(String[] header, String name) {
        for (int i = 0; i < header.length; i++) {
            if (header[i].equals(name)) {
                return i;
            }
        }
        throw new IllegalStateException("Checkpoint CSV missing expected column: " + name);
    }

    private record CheckpointRow(String taskId, String condition, boolean success, boolean answered) {
    }

    // ------------------------------------------------------------------
    // Workspace setup
    // ------------------------------------------------------------------

    private static void resetWorkspace(MockTaskSpec task) throws IOException {
        deleteDirectoryIfExists(WORKSPACE);
        Files.createDirectories(WORKSPACE);

        for (var entry : task.files().entrySet()) {
            Path file = WORKSPACE.resolve(entry.getKey()).normalize();

            Path normalizedWorkspace = WORKSPACE.toAbsolutePath().normalize();
            Path normalizedFile = file.toAbsolutePath().normalize();

            if (!normalizedFile.startsWith(normalizedWorkspace)) {
                throw new IOException("Unsafe task file path: " + entry.getKey());
            }

            Files.createDirectories(file.getParent());
            Files.writeString(file, entry.getValue(), StandardCharsets.UTF_8);
        }

        Files.createDirectories(WORKSPACE.resolve("examples"));
    }

    private static void deleteDirectoryIfExists(Path directory) throws IOException {
        if (!Files.exists(directory)) {
            return;
        }

        try (var walk = Files.walk(directory)) {
            walk.sorted(java.util.Comparator.reverseOrder())
                    .forEach(path -> {
                        try {
                            Files.deleteIfExists(path);
                        } catch (IOException e) {
                            throw new RuntimeException("Failed to delete " + path, e);
                        }
                    });
        }
    }
}
