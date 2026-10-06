/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package org.cas.agentgovernor.experiment;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Random;
import java.util.Set;
import java.util.TreeMap;
import java.util.HexFormat;
import java.security.MessageDigest;

import org.cas.agentgovernor.agents.InducedCostGovernedLLMAgent;
import org.cas.agentgovernor.core.CASGovernor;
import org.cas.agentgovernor.experiment.tasks.MockTaskSet;
import org.cas.agentgovernor.experiment.tasks.MockTaskSpec;
import org.cas.agentgovernor.llm.GeminiLLMClient;
import org.cas.agentgovernor.llm.ApiBudgetExceededException;
import org.cas.agentgovernor.llm.GeminiQuotaExhaustedException;
import org.cas.agentgovernor.llm.LLMClient;
import org.cas.agentgovernor.llm.MockClingingLLM;
import org.cas.agentgovernor.metrics.AgentRunMetrics;
import org.cas.agentgovernor.tools.FileTool;
import org.cas.agentgovernor.tools.ToolRegistry;

/**
 * Real-LLM (Gemini) pilot of the paper-4 control-generated-disturbance test.
 * Runs {@link InducedCostGovernedLLMAgent} over a SMALL induced-cost sweep so
 * a genuine LLM agent -- not the hand-coded stochastic policy of the earlier
 * phases -- is subjected to a governor whose own tightening carries a cost.
 *
 * <h2>Budget discipline</h2>
 * Every episode has up to {@code MAX_STEPS} model decisions; retries are
 * additional request attempts. The persistent budget counts all attempts,
 * including failed requests (not all of which are necessarily billable):
 * <ul>
 *   <li>The grid is deliberately tiny (see constants); at defaults it is
 *   {@code ALPHAS x MODES x tasks x REPEATS} episodes.</li>
 *   <li>A persisted attempt cap is enforced before every SDK request,
 *   including retries. It is not a token or dollar spending limit.</li>
 *   <li>Every completed (task, mode, alpha, repeat) row is appended to a
 *   checkpoint CSV immediately and skipped on a later run, so a quota stop
 *   mid-sweep preserves completed episodes. An interrupted episode is retried
 *   from the start, but its previous API attempts remain charged to the cap.</li>
 *   <li>A daily {@link GeminiQuotaExhaustedException} ends the run gracefully
 *   with a progress report rather than crashing.</li>
 * </ul>
 *
 * <p><b>Run modes.</b> {@code java ... RunGeminiInducedCost mock} runs against
 * {@link MockClingingLLM} and makes ZERO API calls -- use it to validate the
 * pipeline end-to-end for free. {@code java ... RunGeminiInducedCost} (no arg,
 * or {@code gemini}) makes real Gemini calls and requires GOOGLE_API_KEY.
 * Optional {@code --run-id NAME} isolates independent runs; {@code --max-api-calls N}
 * explicitly sets/changes that run's lifetime attempt ceiling. Legacy checkpoints
 * are never adopted automatically. See docs/gemini-induced-runs.md.
 *
 * @author veronique
 */
public class RunGeminiInducedCost {

    private static final String MODEL = "gemini-2.5-flash";
    private static final int MAX_STEPS = 16;
    // Powered rerun: 8 repeats (was 3). At alpha=1 induced firing is a ~0.56
    // per-escalation coin flip, so 3 repeats could not resolve a per-task
    // effect; 8 gives real power for the binary success rate.
    private static final int REPEATS = 8;
    // alpha=0.5 dropped from the pilot grid: it was uninformative (induced cost
    // too weak to engage Gemini; see the pilot analysis). 0.0 stays as the
    // regression anchor.
    private static final double[] ALPHAS = {0.0, 1.0, 2.0};
    private static final double BACKOFF_LAMBDA = 0.3;

    /**
     * Default ceiling on API attempts across all resumptions of one run. Sized for the
     * powered grid (6 tasks x 4 modes x 3 alphas x 8 repeats = 576 episodes,
     * ~3500 calls); the checkpoint makes this safe to run across several
     * opt-in days.
     */
    private static final int MAX_API_CALLS = 6000;

    /** Spacing between calls to respect free-tier RPM (~10/min). */
    private static final long INTER_CALL_DELAY_MILLIS = 7_000L;

    private static final Path OUTPUT_DIR = Path.of("output");

    private record Mode(String name, boolean costAware, double k) {
    }

    // Backoff sensitivity k is swept: stronger feedback is what can bifurcate
    // seeds into a regulated vs suppressed basin (paper 4's transition band),
    // and phase 2 found the gentlest k rescued best -- worth testing on a real
    // model. Same k values as the stochastic phase-2 sweep.
    private static final List<Mode> MODES = List.of(
            new Mode("blind", false, 0.0),
            new Mode("aware_k1.5", true, 1.5),
            new Mode("aware_k3.0", true, 3.0),
            new Mode("aware_k6.0", true, 6.0)
    );

    // Powered rerun: only the COLLAPSE-CAPABLE tasks (escalate AND stay
    // solvable, so the induced cost can lower success and backoff can restore
    // it). T002/E002 proved this in the k-sweep run; E003 is fixed to match;
    // C001-C003 are new clones of the recipe. The unsolvable-but-escalating
    // tasks (T001/E004/E001 -- long answers that never substring-match) are
    // dropped: they only ever showed the block-storm mechanism, not a success
    // delta, and that mechanism is already established.
    private static final Set<String> TASK_SUBSET = Set.of(
            "T002_wrong_folder",
            "E002_wrong_dir_and_ext",
            "E003_confident_wrong_report",
            "C001_wrong_subdir",
            "C002_wrong_ext_dir",
            "C003_wrong_root"
    );

    public static void main(String[] args) throws Exception {
        RunOptions options = RunOptions.parse(args);
        boolean mock = options.mock();
        Path runDirectory = InducedRunState.directory(OUTPUT_DIR, mock ? "mock" : "gemini", options.runId());
        if (!mock && !options.explicitRunId()
                && !Files.exists(runDirectory.resolve("run.properties"))
                && Files.exists(OUTPUT_DIR.resolve("gemini_induced_checkpoint.csv"))) {
            throw new IllegalStateException("Legacy checkpoint preserved at output/gemini_induced_checkpoint.csv. "
                    + "Its provenance and attempt count cannot be inferred. To intentionally START A NEW live run, "
                    + "pass gemini --run-id NAME. No legacy rows will be imported.");
        }

        List<MockTaskSpec> tasks = new ArrayList<>();
        List<MockTaskSpec> allCandidates = new ArrayList<>(MockTaskSet.fileRecoveryTasks());
        allCandidates.addAll(MockTaskSet.escalationProneTasks());
        for (MockTaskSpec t : allCandidates) {
            if (TASK_SUBSET.contains(t.taskId())) {
                tasks.add(t);
            }
        }
        if (tasks.size() != TASK_SUBSET.size()
                || tasks.stream().map(MockTaskSpec::taskId).distinct().count() != TASK_SUBSET.size()) {
            throw new IllegalStateException("Task subset is missing or contains duplicate task IDs.");
        }
        Set<String> expectedKeys = new HashSet<>();
        for (double alpha : ALPHAS) for (Mode mode : MODES) for (MockTaskSpec task : tasks)
            for (int repeat = 0; repeat < REPEATS; repeat++)
                expectedKeys.add(runKey(task.taskId(), mode.name(), alpha, repeat));

        try (InducedRunState run = new InducedRunState(runDirectory, configuration(mock, tasks),
                expectedKeys, options.maxApiCalls(), MAX_API_CALLS)) {
            runSweep(run, mock, tasks);
        }
    }

    private static void runSweep(InducedRunState run, boolean mock, List<MockTaskSpec> tasks) throws Exception {
        int planned = ALPHAS.length * MODES.size() * tasks.size() * REPEATS;
        System.out.println((mock ? "[MOCK] " : "[GEMINI] ") + "Induced-cost LLM pilot: "
                + ALPHAS.length + " alphas x " + MODES.size() + " modes x " + tasks.size()
                + " tasks x " + REPEATS + " repeats = " + planned + " episodes"
                + (mock ? " (0 API calls)." : ", up to " + MAX_STEPS + " decisions each; reserved attempts "
                        + run.attempts() + "/" + run.limit() + "."));
        System.out.println("Checkpoint: " + run.checkpoint().toAbsolutePath());
        int completed = 0;
        int skipped = 0;

        GeminiLLMClient sharedGemini = null;
        try {
            for (double alpha : ALPHAS) {
                for (Mode mode : MODES) {
                    for (MockTaskSpec task : tasks) {
                        for (int repeat = 0; repeat < REPEATS; repeat++) {
                            String key = runKey(task.taskId(), mode.name(), alpha, repeat);
                            if (run.done(key)) {
                                skipped++;
                                continue;
                            }

                            if (!mock && run.attempts() >= run.limit()) {
                                System.out.println(new ApiBudgetExceededException(run.attempts(), run.limit()).getMessage());
                                return;
                            }

                            // Avoid credentials/network-client setup when the run is already complete.
                            if (!mock && sharedGemini == null) sharedGemini = new GeminiLLMClient(MODEL, run::reserveAttempt);

                            LLMClient llm = mock
                                    ? new MockClingingLLM(task.expectedTarget(), task.expectedAnswer())
                                    : sharedGemini;

                            try {
                                // No inter-call delay in mock mode (no rate limit to respect).
                                runOne(run, llm, task, mode, alpha, repeat, mock ? 0L : INTER_CALL_DELAY_MILLIS);
                                completed++;
                                System.out.println("  done: " + key
                                        + (mock ? "" : "  (reserved API attempts: " + run.attempts() + ")"));
                            } catch (ApiBudgetExceededException e) {
                                System.out.println(e.getMessage() + " Incomplete episode not checkpointed: " + key);
                                return;
                            } catch (GeminiQuotaExhaustedException e) {
                                System.out.println("Gemini quota/transient limit hit ("
                                        + (e.isDailyQuotaExhausted() ? "daily" : "per-minute or 5xx, retries exhausted")
                                        + "). Completed " + completed + " new episodes this run, "
                                        + skipped + " already done. Re-run to resume.");
                                return;
                            } catch (Exception e) {
                                // Any other unexpected error (network, parse, etc.): stop
                                // cleanly rather than crash. Every completed episode is
                                // already checkpointed, so re-running resumes with no loss.
                                System.out.println("Unexpected error on " + key + ": " + e
                                        + ". Stopping cleanly; " + completed + " new episodes done, "
                                        + skipped + " already done. Re-run to resume.");
                                return;
                            }
                        }
                    }
                }
            }
        } finally {
            if (sharedGemini != null) sharedGemini.close();
        }

        System.out.println("Pilot complete. New episodes: " + completed + ", skipped (already done): "
                + skipped + (mock ? "." : ", reserved API attempts: " + run.attempts() + "."));
    }

    private static void runOne(InducedRunState run, LLMClient llm, MockTaskSpec task, Mode mode, double alpha, int repeat,
            long delayMillis)
            throws Exception {
        Path workspace = run.workspaceRoot()
                .resolve(task.taskId() + "_" + mode.name() + "_a" + fmt2(alpha) + "_r" + repeat);
        resetWorkspace(task, workspace);

        ToolRegistry tools = new ToolRegistry();
        tools.register(new FileTool(workspace.toString()));

        CASGovernor governor = CASGovernor.forExpectedTarget(task.expectedTarget());

        // Deterministic induced/backoff stream per (task, mode, alpha, repeat).
        long inducedSeed = inducedSeedFor(task.taskId(), mode.name(), alpha, repeat);

        InducedCostGovernedLLMAgent agent = new InducedCostGovernedLLMAgent(
                llm, governor, tools, delayMillis, alpha, new Random(inducedSeed),
                mode.costAware(), BACKOFF_LAMBDA, mode.k());

        agent.run(task.prompt(), MAX_STEPS);

        AgentRunMetrics metrics = AgentRunMetrics.fromHistory(
                "gemini_" + mode.name() + "_a" + fmt2(alpha), agent.history(), task.expectedAnswer());

        int hardBlocks = 0;
        for (var e : agent.history()) {
            if (e.message() != null && e.message().startsWith("HARD_BLOCK:")) {
                hardBlocks++;
            }
        }
        double meanIntensity = agent.regulationIntensityHistory().stream()
                .mapToDouble(Double::doubleValue).average().orElse(0.0);

        deleteDirectoryIfExists(workspace);

        run.append(String.join(",",
                task.taskId(),
                mode.name(),
                fmt2(alpha),
                Integer.toString(repeat),
                Boolean.toString(metrics.success()),
                Boolean.toString(metrics.answered()),
                Integer.toString(metrics.postErrorToolBurden()),
                Integer.toString(hardBlocks),
                Integer.toString(agent.inducedFailureCount()),
                Integer.toString(agent.backoffStepCount()),
                fmt5(meanIntensity)));
    }

    // --- checkpoint helpers ---

    private static String runKey(String taskId, String mode, double alpha, int repeat) {
        return String.join(",", taskId, mode, fmt2(alpha), Integer.toString(repeat));
    }

    /**
     * Induced/backoff RNG seed per (task, mode, alpha, repeat), using the same
     * big-prime accumulator the stochastic harnesses use.
     *
     * <p>NOT {@code runKey(...).hashCode()}: that made consecutive repeats
     * near-identical. String.hashCode() for keys differing only in the trailing
     * "r0"/"r1"/"r2" differs by exactly 1, and {@code new Random(s)} vs
     * {@code new Random(s+1)} yield nearly the same first {@code nextDouble()},
     * so the repeat draws that decide induced firing were effectively the same
     * -- collapsing 3 repeats to ~1 independent sample of the induced RNG.
     *
     * <p>A multiply-accumulate alone is NOT enough here: whichever field varies
     * LAST still lands in the low bits (repeat -> {@code 31*h + repeat} differs
     * by 1 across repeats), reproducing the same near-constant-low-bits problem.
     * So the accumulated value is passed through a splitmix64 avalanche
     * finalizer, which scrambles every bit -- a 1-bit input difference produces
     * a fully different seed, and hence an independent first {@code nextDouble()}
     * per repeat regardless of field order.
     */
    private static long inducedSeedFor(String taskId, String mode, double alpha, int repeat) {
        long h = 1125899906842597L;
        h = 31 * h + taskId.hashCode();
        h = 31 * h + mode.hashCode();
        h = 31 * h + Double.hashCode(alpha);
        h = 31 * h + repeat;
        return splitmix64(h);
    }

    /** splitmix64 finalizer: avalanches every bit so near-equal inputs de-correlate. */
    private static long splitmix64(long z) {
        z = (z ^ (z >>> 30)) * 0xbf58476d1ce4e5b9L;
        z = (z ^ (z >>> 27)) * 0x94d049bb133111ebL;
        return z ^ (z >>> 31);
    }

    // --- workspace helpers (examples/ dir matches the other harnesses) ---

    private static void resetWorkspace(MockTaskSpec task, Path workspace) throws IOException {
        deleteDirectoryIfExists(workspace);
        Files.createDirectories(workspace);
        for (var fileEntry : task.files().entrySet()) {
            Path file = workspace.resolve(fileEntry.getKey()).normalize();
            if (!file.toAbsolutePath().normalize().startsWith(workspace.toAbsolutePath().normalize())) {
                throw new IOException("Unsafe task file path: " + fileEntry.getKey());
            }
            Files.createDirectories(file.getParent());
            Files.writeString(file, fileEntry.getValue(), StandardCharsets.UTF_8);
        }
        Files.createDirectories(workspace.resolve("examples"));
    }

    private static void deleteDirectoryIfExists(Path directory) throws IOException {
        if (!Files.exists(directory)) {
            return;
        }
        try (var walk = Files.walk(directory)) {
            walk.sorted(java.util.Comparator.reverseOrder()).forEach(path -> {
                try {
                    Files.deleteIfExists(path);
                } catch (IOException e) {
                    throw new RuntimeException("Failed to delete " + path, e);
                }
            });
        }
    }

    private static String fmt2(double v) {
        return String.format(Locale.ROOT, "%.2f", v);
    }

    private static String fmt5(double v) {
        return String.format(Locale.ROOT, "%.5f", v);
    }

    record RunOptions(boolean mock, String runId, boolean explicitRunId, Long maxApiCalls) {
        static RunOptions parse(String[] args) {
            boolean mock = false;
            String id = "default";
            boolean explicit = false;
            Long cap = null;
            int i = 0;
            if (args.length > 0 && (args[0].equalsIgnoreCase("mock") || args[0].equalsIgnoreCase("gemini"))) {
                mock = args[0].equalsIgnoreCase("mock");
                i++;
            }
            while (i < args.length) {
                String option = args[i++];
                if (i == args.length) throw new IllegalArgumentException("Missing value for " + option);
                String value = args[i++];
                if (option.equals("--run-id") && !explicit) {
                    id = value;
                    explicit = true;
                } else if (option.equals("--max-api-calls") && cap == null) {
                    cap = Long.parseLong(value);
                    if (cap < 1) throw new IllegalArgumentException("API cap must be positive");
                } else throw new IllegalArgumentException("Unknown or repeated option: " + option);
            }
            InducedRunState.directory(OUTPUT_DIR, mock ? "mock" : "gemini", id);
            return new RunOptions(mock, id, explicit, cap);
        }
    }

    static String configuration(boolean mock, List<MockTaskSpec> tasks) throws Exception {
        // Human-readable, deterministically ordered settings plus exact compiled implementation hashes.
        // Budget ceiling is intentionally separate: explicit increases must not fork the experiment.
        var config = new TreeMap<String, String>();
        config.put("protocol", "induced-v2");
        config.put("backend", mock ? "mock" : "gemini");
        config.put("model", mock ? "MockClingingLLM" : MODEL);
        config.put("sdk", "google-genai:1.57.0;generation=provider-defaults;sdk-attempts=1");
        config.put("max_steps", Integer.toString(MAX_STEPS));
        config.put("repeats", Integer.toString(REPEATS));
        config.put("alphas", java.util.Arrays.toString(ALPHAS));
        config.put("modes", MODES.toString());
        config.put("backoff_lambda", Double.toString(BACKOFF_LAMBDA));
        config.put("inter_call_delay_ms", Long.toString(mock ? 0 : INTER_CALL_DELAY_MILLIS));
        config.put("seed", "task+mode+alpha+repeat;splitmix64;java.util.Random");
        config.put("task_order", tasks.stream().map(MockTaskSpec::taskId).toList().toString());
        for (MockTaskSpec t : tasks) {
            config.put("task." + t.taskId() + ".prompt", t.prompt());
            config.put("task." + t.taskId() + ".target", t.expectedTarget());
            config.put("task." + t.taskId() + ".answer", t.expectedAnswer());
            t.files().forEach((path, contents) -> config.put("task." + t.taskId() + ".file." + path, contents));
        }
        for (Class<?> cls : List.of(RunGeminiInducedCost.class, InducedRunState.class,
                InducedCostGovernedLLMAgent.class, CASGovernor.class, AgentRunMetrics.class,
                GeminiLLMClient.class, MockClingingLLM.class, FileTool.class, ToolRegistry.class,
                org.cas.agentgovernor.core.RegulatorySignal.class,
                org.cas.agentgovernor.core.AgentEvent.class,
                org.cas.agentgovernor.core.ToolDiscovery.class,
                org.cas.agentgovernor.core.ToolAction.class,
                org.cas.agentgovernor.core.ToolResult.class,
                org.cas.agentgovernor.core.GovernorDecision.class,
                org.cas.agentgovernor.core.RegulatoryState.class,
                org.cas.agentgovernor.llm.LLMRequest.class,
                org.cas.agentgovernor.llm.LLMResponse.class)) {
            hashClass(config, cls);
        }
        StringBuilder canonical = new StringBuilder();
        config.forEach((key, value) -> canonical.append(key.length()).append(':').append(key)
                .append(value.length()).append(':').append(value).append('\n'));
        return canonical.toString();
    }

    private static void hashClass(TreeMap<String, String> config, Class<?> cls) throws Exception {
        try (var input = cls.getResourceAsStream("/" + cls.getName().replace('.', '/') + ".class")) {
            if (input == null) throw new IOException("Cannot fingerprint " + cls.getName());
            config.put("class." + cls.getName(), HexFormat.of().formatHex(
                    MessageDigest.getInstance("SHA-256").digest(input.readAllBytes())));
        }
        for (Class<?> nested : cls.getDeclaredClasses()) hashClass(config, nested);
    }
}
