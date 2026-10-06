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
import org.cas.agentgovernor.tools.FlakyFileTool;
import org.cas.agentgovernor.tools.ToolRegistry;

/**
 * Identical in mechanics to {@link RunGeminiTaskComparison} -- same
 * ungoverned/governed/sham_regulation design, same checkpoint-and-resume
 * pattern, same signal-sequence persistence for sham pairing -- but:
 *
 *   1. Runs {@code MockTaskSet.hardTasks()} instead of
 *      {@code MockTaskSet.fileRecoveryTasks()}. The original 8 tasks
 *      turned out to be too easy for Gemini 2.5 Flash: 100% success on
 *      BOTH ungoverned and governed across all 71 real episodes collected
 *      (see the human-graded review of that run), a ceiling effect that
 *      made the governed-vs-ungoverned comparison completely
 *      uninformative on real Gemini data, even though sham_regulation
 *      still showed a large, real gap (real content failures, not
 *      grading artifacts). hardTasks() targets difficulty sources a
 *      genuinely capable model can still be challenged by: content-based
 *      disambiguation among structurally-identical candidates,
 *      multi-file synthesis, a deprecated/stale-file trap, and
 *      larger-scale content filtering -- see MockTaskSet's javadoc on
 *      hardTasks() for the full rationale per task.
 *
 *   2. MAX_STEPS is tighter (8, not 16) -- budget pressure was the other
 *      untested lever. A capable model given a generous budget may simply
 *      brute-force even harder tasks by checking every candidate file,
 *      which would reproduce the same ceiling-effect problem this class
 *      exists to avoid.
 *
 *   3. Uses ENTIRELY SEPARATE output files (checkpoint CSV, signal
 *      sequence directory, aggregate summary, workspace root) from
 *      RunGeminiTaskComparison, all suffixed "_hard". This is deliberate
 *      isolation: the original 71-episode dataset is already collected
 *      and (informally) analyzed, and nothing here should be able to
 *      touch, resume into, or overwrite it. Running this class is
 *      completely independent of and safe alongside the original.
 *
 * See RunGeminiTaskComparison's class doc for the full mechanics
 * (checkpointing, quota handling, sham pairing) -- all of that reasoning
 * applies here unchanged.
 *
 * @author veronique
 */
public class RunGeminiHardTaskComparison {

    private static final Path WORKSPACE = Path.of("test_workspace_hard");
    private static final Path OUTPUT_DIR = Path.of("output");
    private static final Path CHECKPOINT_CSV = OUTPUT_DIR.resolve("gemini_hard_task_comparison_checkpoint.csv");
    private static final Path SIGNAL_SEQUENCE_DIR = OUTPUT_DIR.resolve("gemini_hard_signal_sequences");

    private static final String MODEL = "gemini-2.5-flash";

    private static final long DELAY_MILLIS = 8_000L;
    private static final long PAUSE_BETWEEN_RUNS_MILLIS = 15_000L;

    /** Deliberately tighter than RunGeminiTaskComparison's 16 -- see class doc point 2. */
    private static final int MAX_STEPS = 8;

    /** Repeats per (task, condition). Kept small on purpose -- see RunGeminiTaskComparison's class doc. */
    private static final int REPEATS = 3;

    /**
     * Configuration for the governed_adaptive condition's bounded-retry
     * governor. 2 matches H005's flakyReadFailures exactly -- enough retry
     * budget to get through the flaky primary's exact failure count and
     * succeed via the same path ungoverned uses, while still hard-blocking
     * a genuinely broken/misclassified "transient" failure once the budget
     * is exhausted. See CASGovernor.withBoundedRetry()'s doc.
     */
    private static final int ADAPTIVE_MAX_TRANSIENT_RETRIES = 2;
    private static final Set<String> ADAPTIVE_TRANSIENT_ERROR_TYPES = Set.of("TRANSIENT_READ_ERROR");

    public static void main(String[] args) throws Exception {
        Files.createDirectories(OUTPUT_DIR);
        Files.createDirectories(SIGNAL_SEQUENCE_DIR);

        Set<String> completed = loadCompletedRunKeys();
        System.out.println("Resuming with " + completed.size() + " runs already completed "
                + "(from " + CHECKPOINT_CSV.toAbsolutePath() + ").");

        ensureCheckpointHeader();

        List<MockTaskSpec> tasks = MockTaskSet.hardTasks();

        try (GeminiLLMClient llm = new GeminiLLMClient(MODEL)) {
            try {
                for (MockTaskSpec task : tasks) {
                    for (int repeat = 0; repeat < REPEATS; repeat++) {
                        runOneIfNeeded(llm, task, "ungoverned", repeat, completed);
                        runOneIfNeeded(llm, task, "governed", repeat, completed);
                        runOneIfNeeded(llm, task, "governed_adaptive", repeat, completed);
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
                    System.out.println("exactly where it left off.");
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
     * Builds the correct governor for a task: multi-target (AND semantics,
     * MockTaskSpec.isMultiTarget() / CASGovernor.forExpectedTargets()) for
     * tasks needing every listed file; any-of (OR semantics,
     * MockTaskSpec.hasAcceptableTargets() / CASGovernor.forAnyOfExpectedTargets())
     * for tasks where multiple independent sources each suffice -- this is
     * what H005's flaky-primary-plus-reliable-backup design uses, so the
     * governor recognizes a successful backup read as "useful content
     * recovered" just as validly as the primary; or the common single-target
     * case otherwise.
     */
    private static CASGovernor buildGovernor(MockTaskSpec task) {
        if (task.isMultiTarget()) {
            return CASGovernor.forExpectedTargets(task.requiredTargets());
        }
        if (task.hasAcceptableTargets()) {
            return CASGovernor.forAnyOfExpectedTargets(task.acceptableTargets());
        }
        return CASGovernor.forExpectedTarget(task.expectedTarget());
    }

    /**
     * Same relevance configuration as buildGovernor(), but with bounded
     * retry enabled (see CASGovernor.withBoundedRetry()'s doc). This is
     * expected to be behaviorally IDENTICAL to buildGovernor() on every
     * task except ones using FlakyFileTool (currently only H005): those
     * tasks' failures use errorType "TRANSIENT_READ_ERROR", which isn't
     * produced by any other task's failures (wrong path / wrong extension
     * / etc. all use different errorTypes from plain FileTool), so
     * isBlockedRepeatWithBoundedRetry's "was every prior failure
     * transient-classified" check will be false for all of them, falling
     * through to the same "block immediately" behavior as before. This
     * condition exists specifically to test whether a governor that
     * distinguishes transient from deterministic failures can recover
     * H005's lost ground without giving up any of governed's existing wins
     * elsewhere -- see the class doc.
     */
    private static CASGovernor buildAdaptiveGovernor(MockTaskSpec task) {
        return buildGovernor(task).withBoundedRetry(
                ADAPTIVE_MAX_TRANSIENT_RETRIES,
                ADAPTIVE_TRANSIENT_ERROR_TYPES
        );
    }

    /**
     * Builds this task's tool registry, wrapping FileTool in FlakyFileTool
     * when the task declares flaky-read targets (see MockTaskSpec.hasFlakyReads()).
     */
    private static ToolRegistry buildTools(MockTaskSpec task) {
        ToolRegistry tools = new ToolRegistry();
        FileTool fileTool = new FileTool(WORKSPACE.toString());

        if (task.hasFlakyReads()) {
            tools.register(new FlakyFileTool(fileTool, task.flakyReadFailures()));
        } else {
            tools.register(fileTool);
        }

        return tools;
    }

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
        System.out.println("GEMINI-HARD | TASK " + task.taskId() + " | " + condition + " | repeat " + repeat);
        System.out.println("========================================");

        resetWorkspace(task);

        AgentRunMetrics metrics;

        if ("governed".equals(condition)) {
            ToolRegistry tools = buildTools(task);
            CASGovernor governor = buildGovernor(task);

            GovernedLLMAgent agent = new GovernedLLMAgent(llm, governor, tools, DELAY_MILLIS);
            String answer = agent.run(task.prompt(), MAX_STEPS);

            System.out.println("=== GOVERNED FINAL ANSWER ===");
            System.out.println(answer);

            metrics = AgentRunMetrics.fromHistory("governed_gemini_hard", agent.history(), task.expectedAnswer());

            writeSignalSequence(task.taskId(), repeat, agent.controlSignalHistory());
        } else if ("governed_adaptive".equals(condition)) {
            ToolRegistry tools = buildTools(task);
            CASGovernor governor = buildAdaptiveGovernor(task);

            GovernedLLMAgent agent = new GovernedLLMAgent(llm, governor, tools, DELAY_MILLIS);
            String answer = agent.run(task.prompt(), MAX_STEPS);

            System.out.println("=== GOVERNED_ADAPTIVE FINAL ANSWER ===");
            System.out.println(answer);

            metrics = AgentRunMetrics.fromHistory("governed_adaptive_gemini_hard", agent.history(), task.expectedAnswer());

            // No signal-sequence write here: governed_adaptive doesn't have
            // its own sham pairing yet (out of scope for now -- it's a
            // separate condition, not a replacement for "governed", and
            // sham's whole point is testing whether the LIVE version of
            // "governed" specifically beats a decorrelated copy of itself).
        } else {
            metrics = runUngoverned(llm, task);
        }

        appendCheckpointRow(task.taskId(), condition, repeat, metrics);
        completed.add(key);

        System.out.println("Pausing " + (PAUSE_BETWEEN_RUNS_MILLIS / 1000) + "s before next run...");
        Thread.sleep(PAUSE_BETWEEN_RUNS_MILLIS);
    }

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
        System.out.println("GEMINI-HARD | TASK " + task.taskId() + " | sham_regulation | repeat " + repeat);
        System.out.println("========================================");

        resetWorkspace(task);

        List<RegulatorySignal> shuffled = new ArrayList<>(realizedSequence);
        long shuffleSeed = shuffleSeedFor(task.taskId(), repeat);
        Collections.shuffle(shuffled, new Random(shuffleSeed));

        ToolRegistry tools = buildTools(task);
        CASGovernor governor = buildGovernor(task);

        GovernedLLMAgent agent = new GovernedLLMAgent(llm, governor, tools, DELAY_MILLIS, shuffled);
        String answer = agent.run(task.prompt(), shuffled.size());

        System.out.println("=== SHAM_REGULATION FINAL ANSWER ===");
        System.out.println(answer);

        AgentRunMetrics metrics =
                AgentRunMetrics.fromHistory("sham_regulation_gemini_hard", agent.history(), task.expectedAnswer());

        appendCheckpointRow(task.taskId(), "sham_regulation", repeat, metrics);
        completed.add(key);

        System.out.println("Pausing " + (PAUSE_BETWEEN_RUNS_MILLIS / 1000) + "s before next run...");
        Thread.sleep(PAUSE_BETWEEN_RUNS_MILLIS);
    }

    private static long shuffleSeedFor(String taskId, int repeat) {
        long h = 1125899906842597L;
        h = 31 * h + taskId.hashCode();
        h = 31 * h + repeat;
        h = 31 * h + "sham_shuffle_hard".hashCode();
        return h;
    }

    private static AgentRunMetrics runUngoverned(GeminiLLMClient llm, MockTaskSpec task) throws Exception {
        ToolRegistry tools = buildTools(task);

        UngovernedLLMAgent agent = new UngovernedLLMAgent(
                llm,
                tools,
                DELAY_MILLIS
        );

        String answer = agent.run(task.prompt(), MAX_STEPS);

        System.out.println("=== UNGOVERNED FINAL ANSWER ===");
        System.out.println(answer);

        return AgentRunMetrics.fromHistory("ungoverned_gemini_hard", agent.history(), task.expectedAnswer());
    }

    // ------------------------------------------------------------------
    // Signal sequence persistence (for sham_regulation pairing)
    // ------------------------------------------------------------------

    private static Path signalSequencePath(String taskId, int repeat) {
        String safeName = (taskId + "_repeat" + repeat).replaceAll("[^a-zA-Z0-9_.-]", "_");
        return SIGNAL_SEQUENCE_DIR.resolve(safeName + ".tsv");
    }

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

    private static Set<String> loadCompletedRunKeys() throws IOException {
        Set<String> keys = new HashSet<>();

        if (!Files.exists(CHECKPOINT_CSV)) {
            return keys;
        }

        List<String> lines = Files.readAllLines(CHECKPOINT_CSV, StandardCharsets.UTF_8);

        for (int i = 1; i < lines.size(); i++) {
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
    // Aggregation
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

        for (String condition : List.of("ungoverned", "governed", "governed_adaptive", "sham_regulation")) {
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
                    "gemini_hard",
                    condition,
                    Integer.toString(n),
                    String.format("%.4f", ((double) successes) / n),
                    String.format("%.4f", ((double) answered) / n)
            ));
        }

        Path aggregateCsv = OUTPUT_DIR.resolve("gemini_hard_task_aggregate_summary.csv");
        Files.writeString(
                aggregateCsv,
                String.join(System.lineSeparator(), aggregateRows) + System.lineSeparator(),
                StandardCharsets.UTF_8
        );

        System.out.println("\n========================================");
        System.out.println("GEMINI-HARD AGGREGATE SUMMARY (from checkpoint, may be partial)");
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
