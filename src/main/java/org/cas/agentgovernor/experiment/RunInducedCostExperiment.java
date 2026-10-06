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
import java.util.List;
import java.util.Random;
import java.util.concurrent.Callable;
import java.util.concurrent.CompletionService;
import java.util.concurrent.ExecutorCompletionService;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import org.cas.agentgovernor.agents.InducedCostStochasticToolAgent;
import org.cas.agentgovernor.core.AgentEvent;
import org.cas.agentgovernor.core.CASGovernor;
import org.cas.agentgovernor.experiment.tasks.MockTaskSet;
import org.cas.agentgovernor.experiment.tasks.MockTaskSpec;
import org.cas.agentgovernor.metrics.AgentRunMetrics;
import org.cas.agentgovernor.policy.ActionPolicy;
import org.cas.agentgovernor.policy.PolicyParameters;
import org.cas.agentgovernor.tools.FileTool;
import org.cas.agentgovernor.tools.ToolRegistry;

/**
 * Phase 1 of the discrete test of paper 4's control-generated-disturbance
 * claim ("Noise-Source Structure in IRAM-Omega-Q"). Sweeps the induced-cost
 * scale alpha for the governor's own tightening (see
 * {@link InducedCostStochasticToolAgent} for the exact mechanism) and asks
 * whether the discrete substrate reproduces paper 4's amplification ->
 * stochastic transition band -> suppression arc.
 *
 * <p>Conditions: the {@code impulsive} archetype (the one agent whose burden
 * INCREASES under governance in the main stochastic-policy sweep -- the
 * discrete analog of a controller that destabilizes what it regulates) is
 * the primary target; {@code persistent} is included as a specificity
 * control (its internal-carryover disturbance should dominate, so induced
 * cost matters less). Each is run under both RF ({@code governed}, lag 0)
 * and DF ({@code disturbance_first}, lag 1) so the paired DF-RF contrast can
 * be tracked as a function of alpha -- the widening of that contrast's CI is
 * paper 4's transition-band signature.
 *
 * <p>At alpha = 0 every cell reduces to the ordinary governed / DF run, the
 * intended clean regression check against the main sweep.
 *
 * @author veronique
 */
public class RunInducedCostExperiment {

    private static final Path WORKSPACE_ROOT = Path.of("test_workspace_induced");
    private static final int SEEDS_PER_CONDITION = 2000;
    private static final int[] STEP_BUDGETS = {16, 32};

    // Paper 4's transition band sat near alpha = 21-30 in ITS units; those
    // numbers do not transfer, because the discrete tightening jumps are large
    // (the first NORMAL->DISTURBED escalation has intensity step dg+ ~ 0.75, so
    // dg+^2 ~ 0.56) and the induced probability alpha*dg+^2 saturates by alpha
    // ~ 1.8. What must be matched is the REGIME, not the number: this grid is
    // calibrated (via smoke tests) so the induced probability of the dominant
    // tightening event sweeps 0 -> 1 across it. alpha=0 is the regression check.
    private static final double[] ALPHAS = {0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0};

    private static final int PARALLELISM = Math.max(1, Runtime.getRuntime().availableProcessors());

    /** RF = anticipatory (lag 0); DF = reactive (lag 1). Matches the main sweep. */
    private record Ordering(String name, int lagSteps) {
    }

    private record AgentType(String name, PolicyParameters parameters) {
    }

    public static void main(String[] args) throws Exception {
        List<AgentType> agentTypes = List.of(
                new AgentType("impulsive", PolicyParameters.impulsive()),
                new AgentType("persistent", PolicyParameters.persistent())
        );
        List<Ordering> orderings = List.of(
                new Ordering("RF", 0),
                new Ordering("DF", 1)
        );
        List<MockTaskSpec> tasks = MockTaskSet.fileRecoveryTasks();

        int totalCells = agentTypes.size() * orderings.size() * tasks.size()
                * STEP_BUDGETS.length * ALPHAS.length;
        System.out.println("Induced-cost experiment: " + agentTypes.size() + " agents x "
                + orderings.size() + " orderings x " + tasks.size() + " tasks x "
                + STEP_BUDGETS.length + " step budgets x " + ALPHAS.length + " alphas x "
                + SEEDS_PER_CONDITION + " seeds = " + totalCells + " cells, " + PARALLELISM + " threads.");

        Path outputDir = Path.of("output");
        Files.createDirectories(outputDir);

        List<Callable<CellResult>> cellTasks = new ArrayList<>();
        for (AgentType agentType : agentTypes) {
            for (Ordering ordering : orderings) {
                for (MockTaskSpec task : tasks) {
                    for (int maxSteps : STEP_BUDGETS) {
                        for (double alpha : ALPHAS) {
                            cellTasks.add(buildCellTask(agentType, ordering, task, maxSteps, alpha));
                        }
                    }
                }
            }
        }

        ExecutorService executor = Executors.newFixedThreadPool(PARALLELISM);
        List<RunRecord> allRecords = new ArrayList<>();
        List<String> runRows = new ArrayList<>();
        runRows.add("seed,task_id,agent_type,ordering,lag_steps,alpha,max_steps,"
                + AgentRunMetrics.emptyHeaderWithoutAgentPrefix()
                + ",soft_warnings,hard_blocks,mean_entropy,mean_regulation_intensity,induced_failures");

        try {
            CompletionService<CellResult> completionService = new ExecutorCompletionService<>(executor);
            for (Callable<CellResult> cellTask : cellTasks) {
                completionService.submit(cellTask);
            }
            for (int done = 1; done <= cellTasks.size(); done++) {
                CellResult result = completionService.take().get();
                allRecords.addAll(result.records());
                runRows.addAll(result.runRows());
                if (done % 20 == 0 || done == cellTasks.size()) {
                    System.out.println("  " + done + "/" + cellTasks.size() + " cells complete");
                }
            }
        } finally {
            executor.shutdown();
        }

        deleteDirectoryIfExists(WORKSPACE_ROOT);

        Path runsCsv = outputDir.resolve("induced_cost_runs.csv");
        Files.write(runsCsv, runRows, StandardCharsets.UTF_8);
        System.out.println("Wrote: " + runsCsv.toAbsolutePath());

        List<String> aggregateRows = buildAggregateRows(allRecords);
        Path aggregateCsv = outputDir.resolve("induced_cost_aggregate.csv");
        Files.write(aggregateCsv, aggregateRows, StandardCharsets.UTF_8);
        System.out.println("Wrote: " + aggregateCsv.toAbsolutePath());
    }

    private static Callable<CellResult> buildCellTask(
            AgentType agentType, Ordering ordering, MockTaskSpec task, int maxSteps, double alpha
    ) {
        return () -> {
            List<RunRecord> localRecords = new ArrayList<>();
            List<String> localRunRows = new ArrayList<>();

            String cellDirName = agentType.name() + "_" + ordering.name() + "_"
                    + task.taskId() + "_" + maxSteps + "_a" + fmtAlpha(alpha);
            Path cellWorkspace = WORKSPACE_ROOT.resolve(cellDirName);

            try {
                for (int seed = 0; seed < SEEDS_PER_CONDITION; seed++) {
                    runOne(localRecords, localRunRows, seed, task, agentType, ordering, maxSteps, alpha, cellWorkspace);
                }
            } finally {
                deleteDirectoryIfExists(cellWorkspace);
            }
            return new CellResult(localRecords, localRunRows);
        };
    }

    private static void runOne(
            List<RunRecord> records, List<String> runRows, int seed, MockTaskSpec task,
            AgentType agentType, Ordering ordering, int maxSteps, double alpha, Path workspace
    ) throws IOException {
        resetWorkspace(task, workspace);

        ToolRegistry tools = new ToolRegistry();
        tools.register(new FileTool(workspace.toString()));

        long policySeed = seedFor(seed, task.taskId(), agentType.name(), ordering.name(), maxSteps, alpha, "policy");
        long inducedSeed = seedFor(seed, task.taskId(), agentType.name(), ordering.name(), maxSteps, alpha, "induced");

        ActionPolicy policy = new ActionPolicy(
                agentType.parameters(), new Random(policySeed),
                task.expectedTarget(), task.expectedAnswer());

        CASGovernor governor = CASGovernor.forExpectedTarget(task.expectedTarget());

        InducedCostStochasticToolAgent agent = new InducedCostStochasticToolAgent(
                policy, tools, governor, true, true, ordering.lagSteps(), alpha, new Random(inducedSeed));

        agent.run(task.prompt(), maxSteps);

        String metricName = agentType.name() + "_" + ordering.name() + "_a" + fmtAlpha(alpha) + "_steps" + maxSteps;
        AgentRunMetrics metrics = AgentRunMetrics.fromHistory(metricName, agent.history(), task.expectedAnswer());

        int softWarnings = countMessages(agent.history(), "SOFT_WARNING:");
        int hardBlocks = countMessages(agent.history(), "HARD_BLOCK:");
        double meanEntropy = mean(agent.entropyHistory());
        double meanIntensity = mean(agent.regulationIntensityHistory());
        int inducedFailures = agent.inducedFailureCount();

        records.add(new RunRecord(
                agentType.name(), ordering.name(), task.taskId(), maxSteps, alpha,
                metrics, hardBlocks, meanIntensity, inducedFailures));

        runRows.add(String.join(",",
                Integer.toString(seed),
                task.taskId(),
                agentType.name(),
                ordering.name(),
                Integer.toString(ordering.lagSteps()),
                fmtAlpha(alpha),
                Integer.toString(maxSteps),
                metrics.toCsvRow(),
                Integer.toString(softWarnings),
                Integer.toString(hardBlocks),
                fmt(meanEntropy),
                fmt(meanIntensity),
                Integer.toString(inducedFailures)));
    }

    private static List<String> buildAggregateRows(List<RunRecord> records) {
        List<String> rows = new ArrayList<>();
        rows.add("agent_type,ordering,max_steps,alpha,n,success_rate,answered_rate,"
                + "mean_hard_blocks,mean_post_error_tool_burden,mean_regulation_intensity,"
                + "mean_induced_failures,sd_post_error_tool_burden");

        record CellKey(String agentType, String ordering, int maxSteps, double alpha) {
        }
        java.util.Map<CellKey, List<RunRecord>> grouped = new java.util.LinkedHashMap<>();
        for (RunRecord r : records) {
            grouped.computeIfAbsent(new CellKey(r.agentType(), r.ordering(), r.maxSteps(), r.alpha()),
                    k -> new ArrayList<>()).add(r);
        }

        for (var entry : grouped.entrySet()) {
            List<RunRecord> cell = entry.getValue();
            double successRate = cell.stream().mapToDouble(r -> r.metrics().success() ? 1.0 : 0.0).average().orElse(0.0);
            double answeredRate = cell.stream().mapToDouble(r -> r.metrics().answered() ? 1.0 : 0.0).average().orElse(0.0);
            double meanHardBlocks = cell.stream().mapToInt(RunRecord::hardBlocks).average().orElse(0.0);
            double meanBurden = cell.stream().mapToDouble(r -> r.metrics().postErrorToolBurden()).average().orElse(0.0);
            double meanIntensity = cell.stream().mapToDouble(RunRecord::meanRegulationIntensity).average().orElse(0.0);
            double meanInduced = cell.stream().mapToInt(RunRecord::inducedFailures).average().orElse(0.0);
            double sdBurden = sampleSd(cell.stream().mapToDouble(r -> r.metrics().postErrorToolBurden()).toArray());

            rows.add(String.join(",",
                    entry.getKey().agentType(),
                    entry.getKey().ordering(),
                    Integer.toString(entry.getKey().maxSteps()),
                    fmtAlpha(entry.getKey().alpha()),
                    Integer.toString(cell.size()),
                    fmt(successRate), fmt(answeredRate), fmt(meanHardBlocks),
                    fmt(meanBurden), fmt(meanIntensity), fmt(meanInduced), fmt(sdBurden)));
        }
        return rows;
    }

    private static double sampleSd(double[] values) {
        if (values.length < 2) {
            return 0.0;
        }
        double mean = 0.0;
        for (double v : values) {
            mean += v;
        }
        mean /= values.length;
        double sumSq = 0.0;
        for (double v : values) {
            sumSq += (v - mean) * (v - mean);
        }
        return Math.sqrt(sumSq / (values.length - 1));
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

    private static String fmt(double value) {
        return String.format("%.5f", value);
    }

    private static String fmtAlpha(double alpha) {
        return String.format("%.1f", alpha);
    }

    private static long seedFor(int seed, String taskId, String agentType, String ordering,
            int maxSteps, double alpha, String stream) {
        long h = 1125899906842597L;
        h = 31 * h + seed;
        h = 31 * h + taskId.hashCode();
        h = 31 * h + agentType.hashCode();
        h = 31 * h + ordering.hashCode();
        h = 31 * h + maxSteps;
        h = 31 * h + Double.hashCode(alpha);
        h = 31 * h + stream.hashCode();
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

    private static void resetWorkspace(MockTaskSpec task, Path workspace) throws IOException {
        deleteDirectoryIfExists(workspace);
        Files.createDirectories(workspace);

        for (var fileEntry : task.files().entrySet()) {
            Path file = workspace.resolve(fileEntry.getKey()).normalize();
            Path normalizedWorkspace = workspace.toAbsolutePath().normalize();
            Path normalizedFile = file.toAbsolutePath().normalize();
            if (!normalizedFile.startsWith(normalizedWorkspace)) {
                throw new IOException("Unsafe task file path: " + fileEntry.getKey());
            }
            Files.createDirectories(file.getParent());
            Files.writeString(file, fileEntry.getValue(), StandardCharsets.UTF_8);
        }

        // Must match RunStochasticPolicyExperiment.resetWorkspace exactly, so
        // the alpha=0 regression check reproduces the main sweep's baselines
        // (see the switching-experiment workspace-mismatch bug).
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

    private record RunRecord(
            String agentType, String ordering, String taskId, int maxSteps, double alpha,
            AgentRunMetrics metrics, int hardBlocks, double meanRegulationIntensity, int inducedFailures
    ) {
    }

    private record CellResult(List<RunRecord> records, List<String> runRows) {
    }
}
