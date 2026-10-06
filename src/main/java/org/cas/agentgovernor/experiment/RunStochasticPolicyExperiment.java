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
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicInteger;

import org.cas.agentgovernor.agents.StochasticToolAgent;
import org.cas.agentgovernor.core.AgentEvent;
import org.cas.agentgovernor.core.CASGovernor;
import org.cas.agentgovernor.core.RegulatorySignal;
import org.cas.agentgovernor.core.RegulatoryState;
import org.cas.agentgovernor.experiment.tasks.MockTaskSet;
import org.cas.agentgovernor.experiment.tasks.MockTaskSpec;
import org.cas.agentgovernor.metrics.AgentRunMetrics;
import org.cas.agentgovernor.policy.ActionPolicy;
import org.cas.agentgovernor.policy.PolicyParameters;
import org.cas.agentgovernor.tools.FileTool;
import org.cas.agentgovernor.tools.ToolRegistry;

/**
 * Stochastic-policy sweep across seven conditions per (agent type, task,
 * step budget, seed):
 *
 *   ungoverned        - static base weights all episode, no governor enforcement.
 *   reweight_only     - ActionPolicy weights are modulated live by the regulatory
 *                       signal, but the governor never warns or hard-blocks a
 *                       proposed action/final-answer.
 *   block_only        - weights are never modulated, but the governor can warn on
 *                       or hard-block (e.g. repeat actions, premature answers,
 *                       RELEASED-state tool use).
 *   governed          - both mechanisms active together, applied WITHOUT lag
 *                       (regulation-first / RF ordering: the controller sees
 *                       every disturbance before the very next action commits).
 *   disturbance_first - identical mechanism strength to "governed" (reweight
 *                       AND enforceBlocking both on), but the signal driving
 *                       that mechanism is delayed by 1 real tool-call outcome
 *                       (disturbance-first / DF ordering). Isolates causal-
 *                       ordering effects from mechanism-strength effects --
 *                       see StochasticToolAgent's class doc and Ziegler,
 *                       arXiv:2603.16020.
 *   sham_regulation   - THE PLACEBO CONTROL. Identical mechanism strength to
 *                       "governed" (reweight AND enforceBlocking both on),
 *                       but the per-step regulatory state applied is drawn
 *                       from a SHUFFLED copy of that same seed's own
 *                       "governed" run's realized state sequence -- same
 *                       multiset of interventions (matched total strength
 *                       and frequency), decorrelated from when disturbances
 *                       actually occurred in this run.
 *   static_good       - base weights are pre-modulated ONCE (as if permanently in
 *                       RECOVERING state) and then held fixed for the whole
 *                       episode; no live signal consultation, no blocking.
 *
 * Rationale: see prereg_cas_governor.md and its addenda for the full
 * scientific rationale behind each condition.
 *
 * PARALLELIZATION: work is partitioned into one task per (agent type, task,
 * step budget) combination -- 5 x 8 x 4 = 160 tasks, each running its own
 * full inner loop over all SEEDS_PER_CONDITION seeds and all 7 conditions
 * sequentially. This granularity was chosen deliberately:
 *
 *   - Each task is a large, roughly-equal chunk of work (SEEDS_PER_CONDITION
 *     x 7 episodes), so thread-pool scheduling overhead is negligible and
 *     load balances well across however many cores are available.
 *   - Every episode already constructs its own fresh ActionPolicy,
 *     CASGovernor, ToolRegistry, FileTool, and Random -- the ONLY shared
 *     mutable state across the original single-threaded version was the
 *     filesystem workspace directory (every episode reset and reused the
 *     SAME "test_workspace" path). Running that concurrently without
 *     changing anything else would silently corrupt results (one thread
 *     deleting/rewriting files while another thread's FileTool reads them
 *     mid-episode). Each of the 160 tasks below gets its OWN workspace
 *     subdirectory (derived deterministically from its agent type, task
 *     id, and step budget), so no two concurrently-running tasks ever
 *     touch the same path.
 *   - Each task accumulates its results into its own LOCAL lists (never
 *     touching a shared List during the parallel phase) and returns them
 *     as a CellResult; only after every task completes are the 160
 *     per-task result lists merged, sequentially, on the calling thread.
 *     This avoids any need to synchronize on the hot path at all.
 *   - Per-episode RNG seeding is still a pure deterministic function of
 *     (seed, taskId, agentType, conditionLabel, maxSteps) -- see seedFor()
 *     -- so results are bit-for-bit reproducible regardless of thread
 *     scheduling or how many cores are available.
 *
 * PARALLELISM defaults to Runtime.getRuntime().availableProcessors().
 * This workload is I/O-heavy (filesystem resets/reads dominate wall time
 * far more than the in-memory policy logic), so it may be worth
 * experimenting with a pool size somewhat larger than your core count if
 * you have fast storage with room to hide latency -- this default is a
 * safe, conservative starting point, not a claim that it's optimal.
 *
 * @author veronique
 */
public class RunStochasticPolicyExperiment {

    private static final Path WORKSPACE_ROOT = Path.of("test_workspace");
    private static final Path OUTPUT_DIR = Path.of("output");

    private static final List<Integer> STEP_BUDGETS = List.of(4, 8, 16, 32);
    private static final int SEEDS_PER_CONDITION = 2000;

    private static final int PARALLELISM = Math.max(1, Runtime.getRuntime().availableProcessors());

    public static void main(String[] args) throws Exception {
        Files.createDirectories(OUTPUT_DIR);

        List<AgentType> agentTypes = List.of(
                new AgentType("exploratory", PolicyParameters.exploratory()),
                new AgentType("clinging", PolicyParameters.clinging()),
                new AgentType("persistent", PolicyParameters.persistent()),
                new AgentType("impulsive", PolicyParameters.impulsive()),
                new AgentType("noisy", PolicyParameters.noisy())
        );

        // Conditions handled by the simple single-episode-per-condition path.
        // "governed" is deliberately NOT here -- it's run via
        // runGovernedThenSham() below, together with its paired sham run,
        // so sham can reuse governed's realized control-signal sequence.
        List<RunSpec> genericConditions = List.of(
                new RunSpec("ungoverned", false, false, false, 0),
                new RunSpec("reweight_only", true, false, false, 0),
                new RunSpec("block_only", false, true, false, 0),
                new RunSpec("disturbance_first", true, true, false, 1),
                new RunSpec("static_good", false, false, true, 0)
        );
        RunSpec governedSpec = new RunSpec("governed", true, true, false, 0);

        // Full condition list, in report order, used only for aggregation/
        // printing -- not for driving the run loop.
        List<RunSpec> allConditionsForReporting = List.of(
                new RunSpec("ungoverned", false, false, false, 0),
                new RunSpec("reweight_only", true, false, false, 0),
                new RunSpec("block_only", false, true, false, 0),
                governedSpec,
                new RunSpec("disturbance_first", true, true, false, 1),
                new RunSpec("sham_regulation", true, true, false, 0),
                new RunSpec("static_good", false, false, true, 0)
        );

        List<MockTaskSpec> tasks = MockTaskSet.fileRecoveryTasks();
        int totalCells = agentTypes.size() * tasks.size() * STEP_BUDGETS.size();

        System.out.println("Running " + totalCells + " (agent_type, task, max_steps) cells across "
                + PARALLELISM + " threads (" + SEEDS_PER_CONDITION + " seeds x 7 conditions each)...");

        List<Callable<CellResult>> cellTasks = new ArrayList<>();
        for (AgentType agentType : agentTypes) {
            for (MockTaskSpec task : tasks) {
                for (int maxSteps : STEP_BUDGETS) {
                    cellTasks.add(buildCellTask(agentType, task, maxSteps, genericConditions, governedSpec));
                }
            }
        }

        List<RunRecord> records = new ArrayList<>();
        List<String> runRows = new ArrayList<>();
        runRows.add("seed,task_id,agent_type,condition,max_steps,"
                + AgentRunMetrics.emptyHeaderWithoutAgentPrefix()
                + ",soft_warnings,hard_blocks,mean_entropy,entropy_chi");

        AtomicInteger completed = new AtomicInteger(0);
        ExecutorService executor = Executors.newFixedThreadPool(PARALLELISM);

        try {
            List<Future<CellResult>> futures = new ArrayList<>();
            for (Callable<CellResult> cellTask : cellTasks) {
                futures.add(executor.submit(() -> {
                    CellResult result = cellTask.call();
                    int done = completed.incrementAndGet();
                    if (done % 10 == 0 || done == totalCells) {
                        System.out.println("  completed " + done + "/" + totalCells + " cells");
                    }
                    return result;
                }));
            }

            for (Future<CellResult> future : futures) {
                CellResult result = getUnwrapped(future);
                records.addAll(result.records());
                runRows.addAll(result.runRows());
            }
        } finally {
            executor.shutdown();
        }

        // Clean up all per-cell workspace subdirectories now that every
        // task has finished with them.
        deleteDirectoryIfExists(WORKSPACE_ROOT);

        Path runsCsv = OUTPUT_DIR.resolve("stochastic_policy_runs.csv");
        Files.writeString(
                runsCsv,
                String.join(System.lineSeparator(), runRows) + System.lineSeparator(),
                StandardCharsets.UTF_8
        );

        List<String> aggregateRows = buildAggregateRows(records, allConditionsForReporting);

        Path aggregateCsv = OUTPUT_DIR.resolve("stochastic_policy_aggregate.csv");
        Files.writeString(
                aggregateCsv,
                String.join(System.lineSeparator(), aggregateRows) + System.lineSeparator(),
                StandardCharsets.UTF_8
        );

        printAggregateSummary(aggregateRows);

        System.out.println("\nWrote: ");
        System.out.println(runsCsv.toAbsolutePath());
        System.out.println(aggregateCsv.toAbsolutePath());
    }

    /** Unwraps ExecutionException so callers see the real underlying failure. */
    private static CellResult getUnwrapped(Future<CellResult> future) throws Exception {
        try {
            return future.get();
        } catch (ExecutionException e) {
            Throwable cause = e.getCause();
            if (cause instanceof IOException io) {
                throw io;
            }
            if (cause instanceof RuntimeException re) {
                throw re;
            }
            throw e;
        }
    }

    /**
     * Builds one unit of parallel work: running every seed and every
     * condition for a single (agent type, task, step budget) cell, using
     * a workspace subdirectory unique to that cell so it can safely run
     * concurrently with every other cell's task.
     */
    private static Callable<CellResult> buildCellTask(
            AgentType agentType,
            MockTaskSpec task,
            int maxSteps,
            List<RunSpec> genericConditions,
            RunSpec governedSpec
    ) {
        Path cellWorkspace = WORKSPACE_ROOT.resolve(cellDirName(agentType, task, maxSteps));

        return () -> {
            List<RunRecord> localRecords = new ArrayList<>();
            List<String> localRunRows = new ArrayList<>();

            for (int seed = 0; seed < SEEDS_PER_CONDITION; seed++) {
                for (RunSpec spec : genericConditions) {
                    runOne(localRecords, localRunRows, seed, task, agentType, spec, maxSteps, cellWorkspace);
                }
                runGovernedThenSham(localRecords, localRunRows, seed, task, agentType, governedSpec, maxSteps, cellWorkspace);
            }

            return new CellResult(localRecords, localRunRows);
        };
    }

    /** Filesystem-safe, collision-free directory name for one cell's isolated workspace. */
    private static String cellDirName(AgentType agentType, MockTaskSpec task, int maxSteps) {
        String raw = agentType.name() + "_" + task.taskId() + "_steps" + maxSteps;
        return raw.replaceAll("[^a-zA-Z0-9_.-]", "_");
    }

    private static void runOne(
            List<RunRecord> records,
            List<String> runRows,
            int seed,
            MockTaskSpec task,
            AgentType agentType,
            RunSpec spec,
            int maxSteps,
            Path workspace
    ) throws IOException {
        long seedValue = seedFor(seed, task.taskId(), agentType.name(), spec.label(), maxSteps);
        StochasticToolAgent agent = runOneEpisode(task, agentType, spec, maxSteps, seedValue, null, workspace);
        recordRun(records, runRows, seed, task, agentType, spec.label(), maxSteps, agent);
    }

    /**
     * Runs "governed" for this seed, then builds and runs its paired
     * "sham_regulation" episode from governed's own realized per-step
     * control-signal sequence, shuffled. See class doc and
     * StochasticToolAgent's class doc for the full rationale.
     */
    private static void runGovernedThenSham(
            List<RunRecord> records,
            List<String> runRows,
            int seed,
            MockTaskSpec task,
            AgentType agentType,
            RunSpec governedSpec,
            int maxSteps,
            Path workspace
    ) throws IOException {
        long governedSeedValue = seedFor(seed, task.taskId(), agentType.name(), governedSpec.label(), maxSteps);
        StochasticToolAgent governedAgent = runOneEpisode(task, agentType, governedSpec, maxSteps, governedSeedValue, null, workspace);
        recordRun(records, runRows, seed, task, agentType, governedSpec.label(), maxSteps, governedAgent);

        List<RegulatorySignal> realizedSequence = governedAgent.controlSignalHistory();

        if (realizedSequence.isEmpty()) {
            // governed produced a final answer without ever consulting the
            // control signal (e.g. immediate hasSuccessfulExpectedRead exit) --
            // nothing to shuffle. Skip the paired sham run for this seed rather
            // than record a degenerate zero-step episode.
            return;
        }

        List<RegulatorySignal> shuffled = new ArrayList<>(realizedSequence);
        long shuffleSeedValue = seedFor(seed, task.taskId(), agentType.name(), "sham_shuffle", maxSteps);
        Collections.shuffle(shuffled, new Random(shuffleSeedValue));

        RunSpec shamSpec = new RunSpec("sham_regulation", true, true, false, 0);
        long shamSeedValue = seedFor(seed, task.taskId(), agentType.name(), shamSpec.label(), maxSteps);
        StochasticToolAgent shamAgent = runOneEpisode(task, agentType, shamSpec, shuffled.size(), shamSeedValue, shuffled, workspace);
        recordRun(records, runRows, seed, task, agentType, shamSpec.label(), maxSteps, shamAgent);
    }

    /**
     * Constructs a fresh policy and governor against the given (cell-local)
     * workspace, and runs one simulated episode. If
     * {@code scriptedControlSignals} is non-null, the agent runs in sham
     * mode and the effective step budget is exactly
     * {@code scriptedControlSignals.size()} regardless of {@code maxSteps};
     * otherwise {@code maxSteps} is used as-is.
     */
    private static StochasticToolAgent runOneEpisode(
            MockTaskSpec task,
            AgentType agentType,
            RunSpec spec,
            int maxSteps,
            long seedValue,
            List<RegulatorySignal> scriptedControlSignals,
            Path workspace
    ) throws IOException {
        resetWorkspace(task, workspace);

        ToolRegistry tools = new ToolRegistry();
        tools.register(new FileTool(workspace.toString()));

        Random random = new Random(seedValue);

        PolicyParameters baseParameters = spec.useStaticGoodWeights()
                ? staticGoodWeights(agentType.parameters())
                : agentType.parameters();

        ActionPolicy policy = new ActionPolicy(
                baseParameters,
                random,
                task.expectedTarget(),
                task.expectedAnswer()
        );

        CASGovernor governor = CASGovernor.forExpectedTarget(task.expectedTarget());

        StochasticToolAgent agent = new StochasticToolAgent(
                policy,
                tools,
                governor,
                spec.reweight(),
                spec.enforceBlocking(),
                spec.regulationLagSteps(),
                scriptedControlSignals
        );

        int effectiveMaxSteps = scriptedControlSignals != null ? scriptedControlSignals.size() : maxSteps;
        agent.run(task.prompt(), effectiveMaxSteps);

        return agent;
    }

    private static void recordRun(
            List<RunRecord> records,
            List<String> runRows,
            int seed,
            MockTaskSpec task,
            AgentType agentType,
            String condition,
            int maxSteps,
            StochasticToolAgent agent
    ) {
        String metricName = agentType.name() + "_" + condition + "_steps" + maxSteps;
        AgentRunMetrics metrics = AgentRunMetrics.fromHistory(metricName, agent.history(), task.expectedAnswer());

        int softWarnings = countMessages(agent.history(), "SOFT_WARNING:");
        int hardBlocks = countMessages(agent.history(), "HARD_BLOCK:");

        double meanEntropy = mean(agent.entropyHistory());
        double entropyChi = temporalVariance(agent.entropyHistory());

        records.add(new RunRecord(
                seed,
                task.taskId(),
                agentType.name(),
                condition,
                maxSteps,
                metrics,
                softWarnings,
                hardBlocks,
                meanEntropy,
                entropyChi
        ));

        runRows.add(String.join(",",
                Integer.toString(seed),
                task.taskId(),
                agentType.name(),
                condition,
                Integer.toString(maxSteps),
                metrics.toCsvRow(),
                Integer.toString(softWarnings),
                Integer.toString(hardBlocks),
                fmt(meanEntropy),
                fmt(entropyChi)
        ));
    }

    private static double mean(List<Double> values) {
        if (values.isEmpty()) {
            return 0.0;
        }

        double sum = 0.0;
        for (double v : values) {
            sum += v;
        }
        return sum / values.size();
    }

    /**
     * chi: temporal variance of entropy WITHIN one episode's step-by-step
     * entropy trace -- matching EntropySusceptibilityTracker.variance() in
     * the companion IRAM-Omega-Q continuous-formalism project exactly
     * (sample variance, n-1 denominator, computed over one trajectory's
     * own history). This is a real, literal susceptibility analogue, not
     * a metaphor: same formula, same role (an operational criticality
     * indicator), applied to the discrete policy's real entropy trace
     * instead of a continuous quantum-like state's.
     */
    private static double temporalVariance(List<Double> values) {
        int n = values.size();
        if (n < 2) {
            return 0.0;
        }

        double mean = mean(values);
        double sumSq = 0.0;
        for (double v : values) {
            double d = v - mean;
            sumSq += d * d;
        }
        return sumSq / (n - 1);
    }

    /**
     * Weights modulated once, as if the episode were permanently in
     * RECOVERING state, then held fixed. Reuses the exact same multipliers
     * ActionPolicy applies dynamically (via the shared static modulate()
     * utility) so this baseline is "the same good weights" rather than a
     * separately hand-tuned profile that could differ from what governed
     * agents actually converge on.
     */
    private static PolicyParameters staticGoodWeights(PolicyParameters base) {
        RegulatorySignal alwaysRecovering = new RegulatorySignal(
                RegulatoryState.RECOVERING,
                0,
                "Static-good baseline: fixed recovery-shaped weights for the whole episode."
        );

        return ActionPolicy.modulate(base, alwaysRecovering);
    }

    private static long seedFor(
            int seed,
            String taskId,
            String agentType,
            String conditionLabel,
            int maxSteps
    ) {
        long h = 1125899906842597L;
        h = 31 * h + seed;
        h = 31 * h + taskId.hashCode();
        h = 31 * h + agentType.hashCode();
        h = 31 * h + conditionLabel.hashCode();
        h = 31 * h + maxSteps;
        return h;
    }

    private static int countMessages(List<AgentEvent> history, String prefix) {
        int count = 0;

        for (AgentEvent event : history) {
            if (event.message() != null && event.message().startsWith(prefix)) {
                count++;
            }
        }

        return count;
    }

    private static List<String> buildAggregateRows(List<RunRecord> records, List<RunSpec> conditions) {
        List<String> rows = new ArrayList<>();

        rows.add("agent_type,condition,max_steps,n,success_rate,answered_rate,"
                + "mean_tool_calls,sem_tool_calls,"
                + "mean_failed_tool_calls,sem_failed_tool_calls,"
                + "mean_repeated_failed_actions,sem_repeated_failed_actions,"
                + "mean_post_error_tool_burden,sem_post_error_tool_burden,"
                + "mean_regulatory_signals,mean_recovering_signals,mean_stuck_signals,"
                + "mean_soft_warnings,mean_hard_blocks,"
                + "mean_entropy,sem_entropy,mean_entropy_chi,sem_entropy_chi");

        List<String> agentTypes = records.stream()
                .map(RunRecord::agentType)
                .distinct()
                .toList();

        for (String agentType : agentTypes) {
            for (int maxSteps : STEP_BUDGETS) {
                for (RunSpec spec : conditions) {
                    addAggregate(rows, records, agentType, spec.label(), maxSteps);
                }
            }
        }

        return rows;
    }

    private static void addAggregate(
            List<String> rows,
            List<RunRecord> records,
            String agentType,
            String condition,
            int maxSteps
    ) {
        List<RunRecord> selected = records.stream()
                .filter(r -> r.agentType().equals(agentType))
                .filter(r -> r.condition().equals(condition))
                .filter(r -> r.maxSteps() == maxSteps)
                .toList();

        if (selected.isEmpty()) {
            return;
        }

        rows.add(String.join(",",
                agentType,
                condition,
                Integer.toString(maxSteps),
                Integer.toString(selected.size()),
                fmt(meanSuccess(selected)),
                fmt(meanAnswered(selected)),
                fmt(mean(selected, MetricField.TOTAL_TOOL_CALLS)),
                fmt(sem(selected, MetricField.TOTAL_TOOL_CALLS)),
                fmt(mean(selected, MetricField.FAILED_TOOL_CALLS)),
                fmt(sem(selected, MetricField.FAILED_TOOL_CALLS)),
                fmt(mean(selected, MetricField.REPEATED_FAILED_ACTIONS)),
                fmt(sem(selected, MetricField.REPEATED_FAILED_ACTIONS)),
                fmt(mean(selected, MetricField.POST_ERROR_TOOL_BURDEN)),
                fmt(sem(selected, MetricField.POST_ERROR_TOOL_BURDEN)),
                fmt(mean(selected, MetricField.REGULATORY_SIGNALS)),
                fmt(mean(selected, MetricField.RECOVERING_SIGNALS)),
                fmt(mean(selected, MetricField.STUCK_SIGNALS)),
                fmt(meanExtra(selected, ExtraField.SOFT_WARNINGS)),
                fmt(meanExtra(selected, ExtraField.HARD_BLOCKS)),
                fmt(meanOfEntropy(selected)),
                fmt(semOfEntropy(selected)),
                fmt(meanOfEntropyChi(selected)),
                fmt(semOfEntropyChi(selected))
        ));
    }

    private static double meanOfEntropy(List<RunRecord> records) {
        double sum = 0.0;
        for (RunRecord record : records) {
            sum += record.meanEntropy();
        }
        return sum / records.size();
    }

    private static double semOfEntropy(List<RunRecord> records) {
        if (records.size() <= 1) {
            return 0.0;
        }
        double mean = meanOfEntropy(records);
        double sumSq = 0.0;
        for (RunRecord record : records) {
            double d = record.meanEntropy() - mean;
            sumSq += d * d;
        }
        double variance = sumSq / (records.size() - 1);
        return Math.sqrt(variance / records.size());
    }

    /**
     * Mean, across seeds, of each episode's OWN within-episode entropy
     * variance (chi) -- i.e. mean of chi, not chi of means. This is the
     * quantity directly comparable to EntropySusceptibilityTracker's
     * output in the companion continuous-formalism project, aggregated
     * the same way a phase-diagram sweep would average chi across many
     * trajectories at a fixed parameter setting.
     */
    private static double meanOfEntropyChi(List<RunRecord> records) {
        double sum = 0.0;
        for (RunRecord record : records) {
            sum += record.entropyChi();
        }
        return sum / records.size();
    }

    private static double semOfEntropyChi(List<RunRecord> records) {
        if (records.size() <= 1) {
            return 0.0;
        }
        double mean = meanOfEntropyChi(records);
        double sumSq = 0.0;
        for (RunRecord record : records) {
            double d = record.entropyChi() - mean;
            sumSq += d * d;
        }
        double variance = sumSq / (records.size() - 1);
        return Math.sqrt(variance / records.size());
    }

    private static double meanSuccess(List<RunRecord> records) {
        int count = 0;

        for (RunRecord record : records) {
            if (record.metrics().success()) {
                count++;
            }
        }

        return ((double) count) / records.size();
    }

    private static double meanAnswered(List<RunRecord> records) {
        int count = 0;

        for (RunRecord record : records) {
            if (record.metrics().answered()) {
                count++;
            }
        }

        return ((double) count) / records.size();
    }

    private static double mean(List<RunRecord> records, MetricField field) {
        double sum = 0.0;

        for (RunRecord record : records) {
            sum += value(record.metrics(), field);
        }

        return sum / records.size();
    }

    private static double sem(List<RunRecord> records, MetricField field) {
        if (records.size() <= 1) {
            return 0.0;
        }

        double mean = mean(records, field);
        double sumSq = 0.0;

        for (RunRecord record : records) {
            double d = value(record.metrics(), field) - mean;
            sumSq += d * d;
        }

        double variance = sumSq / (records.size() - 1);
        return Math.sqrt(variance / records.size());
    }

    private static double meanExtra(List<RunRecord> records, ExtraField field) {
        double sum = 0.0;

        for (RunRecord record : records) {
            sum += switch (field) {
                case SOFT_WARNINGS -> record.softWarnings();
                case HARD_BLOCKS -> record.hardBlocks();
            };
        }

        return sum / records.size();
    }

    private static double value(AgentRunMetrics metric, MetricField field) {
        return switch (field) {
            case TOTAL_TOOL_CALLS -> metric.totalToolCalls();
            case FAILED_TOOL_CALLS -> metric.failedToolCalls();
            case REPEATED_FAILED_ACTIONS -> metric.repeatedFailedActions();
            case POST_ERROR_TOOL_BURDEN -> metric.postErrorToolBurden();
            case REGULATORY_SIGNALS -> metric.regulatorySignals();
            case RECOVERING_SIGNALS -> metric.recoveringSignals();
            case STUCK_SIGNALS -> metric.stuckSignals();
        };
    }

    private static String fmt(double value) {
        return String.format("%.5f", value);
    }

    private static void printAggregateSummary(List<String> aggregateRows) {
        System.out.println("\n========================================");
        System.out.println("STOCHASTIC POLICY AGGREGATE SUMMARY");
        System.out.println("========================================");

        for (String row : aggregateRows) {
            System.out.println(row);
        }
    }

    /**
     * @param workspace the cell-local workspace directory (NOT the shared
     *                   WORKSPACE_ROOT) -- see class doc's PARALLELIZATION
     *                   section for why this must be per-cell, not global.
     */
    private static void resetWorkspace(MockTaskSpec task, Path workspace) throws IOException {
        deleteDirectoryIfExists(workspace);
        Files.createDirectories(workspace);

        for (var entry : task.files().entrySet()) {
            Path file = workspace.resolve(entry.getKey()).normalize();

            Path normalizedWorkspace = workspace.toAbsolutePath().normalize();
            Path normalizedFile = file.toAbsolutePath().normalize();

            if (!normalizedFile.startsWith(normalizedWorkspace)) {
                throw new IOException("Unsafe task file path: " + entry.getKey());
            }

            Files.createDirectories(file.getParent());
            Files.writeString(file, entry.getValue(), StandardCharsets.UTF_8);
        }

        Files.createDirectories(workspace.resolve("examples"));
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

    private record AgentType(
            String name,
            PolicyParameters parameters
    ) {
    }

    /**
     * @param label                CSV/aggregate condition name.
     * @param reweight             whether ActionPolicy modulates weights by the live signal.
     * @param enforceBlocking      whether the governor can warn/block proposed actions.
     * @param useStaticGoodWeights whether to replace the agent-type's base weights with
     *                             a once-computed, permanently-RECOVERING-shaped profile
     *                             (mutually exclusive in practice with reweight/enforceBlocking,
     *                             both of which are false for this condition).
     * @param regulationLagSteps   see {@code StochasticToolAgent}'s class doc. 0 = regulation-
     *                             first (RF, the default for every condition except
     *                             disturbance_first); &gt;0 = disturbance-first (DF), the signal
     *                             driving reweight/enforceBlocking is delayed by this many real
     *                             tool-call outcomes relative to the true history. Unused (ignored)
     *                             for sham_regulation, which drives its signal from a scripted
     *                             sequence instead -- see runGovernedThenSham().
     */
    private record RunSpec(
            String label,
            boolean reweight,
            boolean enforceBlocking,
            boolean useStaticGoodWeights,
            int regulationLagSteps
    ) {
    }

    /**
     * @param meanEntropy mean Shannon entropy (nats) of the policy's sampling
     *                    distribution across this episode's steps -- see
     *                    ActionPolicy.PolicyDecision#entropy() for what this
     *                    literally is.
     * @param entropyChi  temporal variance of that same per-step entropy trace
     *                    within this one episode -- see
     *                    RunStochasticPolicyExperiment#temporalVariance() for
     *                    why this is a real susceptibility analogue, not a
     *                    metaphor.
     */
    private record RunRecord(
            int seed,
            String taskId,
            String agentType,
            String condition,
            int maxSteps,
            AgentRunMetrics metrics,
            int softWarnings,
            int hardBlocks,
            double meanEntropy,
            double entropyChi
    ) {
    }

    /** Local, per-cell accumulation returned by each parallel task -- never shared/mutated across threads. */
    private record CellResult(
            List<RunRecord> records,
            List<String> runRows
    ) {
    }

    private enum MetricField {
        TOTAL_TOOL_CALLS,
        FAILED_TOOL_CALLS,
        REPEATED_FAILED_ACTIONS,
        POST_ERROR_TOOL_BURDEN,
        REGULATORY_SIGNALS,
        RECOVERING_SIGNALS,
        STUCK_SIGNALS
    }

    private enum ExtraField {
        SOFT_WARNINGS,
        HARD_BLOCKS
    }
}
