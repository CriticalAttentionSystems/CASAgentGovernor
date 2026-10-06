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
 * Phase 2 of the discrete test of paper 4's control-generated-disturbance
 * claim. Runs the induced-cost alpha sweep under a cost-BLIND governor (phase
 * 1's behavior) alongside several cost-AWARE governors that back off when
 * their own tightening has been generating failures (see
 * {@link InducedCostStochasticToolAgent}'s cost-aware mode). The head-to-head
 * design asks two questions paper 4 raises:
 *
 * <ol>
 *   <li><b>Suppression / transition band.</b> Does cost-aware backoff produce
 *   paper 4's graceful suppression (mean regulation intensity falls as alpha
 *   grows, instead of staying pinned) and a stochastic transition band
 *   (seeds splitting into a regulated basin and a suppressed basin), neither
 *   of which the cost-blind governor showed?</li>
 *   <li><b>Rescue.</b> Does letting the governor back off recover the task
 *   success that the cost-blind governor destroyed at high alpha, or is the
 *   underlying regulation genuinely needed (so backoff just trades one
 *   failure mode for another)?</li>
 * </ol>
 *
 * <p>Backoff sensitivity {@code k} is swept ({@code 1.5, 3.0, 6.0}) because
 * stronger feedback is what can bifurcate seeds into two basins; a fixed EWMA
 * weight lambda=0.3 is used throughout. The seed does NOT include the governor
 * mode, so every mode sees the same policy/induced streams per
 * (seed, task, ordering, alpha) up to the point their differing random draws
 * desync -- the fairest available cross-mode comparison.
 *
 * @author veronique
 */
public class RunCostAwareInducedExperiment {

    private static final Path WORKSPACE_ROOT = Path.of("test_workspace_costaware");
    private static final int SEEDS_PER_CONDITION = 2000;
    private static final int[] STEP_BUDGETS = {16, 32};
    private static final double[] ALPHAS = {0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0};
    private static final double BACKOFF_LAMBDA = 0.3;
    private static final int PARALLELISM = Math.max(1, Runtime.getRuntime().availableProcessors());

    private record GovernorMode(String name, boolean costAware, double lambda, double k) {
    }

    private record Ordering(String name, int lagSteps) {
    }

    private record AgentType(String name, PolicyParameters parameters) {
    }

    private static final List<GovernorMode> MODES = List.of(
            new GovernorMode("blind", false, 0.0, 0.0),
            new GovernorMode("aware_k1.5", true, BACKOFF_LAMBDA, 1.5),
            new GovernorMode("aware_k3.0", true, BACKOFF_LAMBDA, 3.0),
            new GovernorMode("aware_k6.0", true, BACKOFF_LAMBDA, 6.0)
    );

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

        int totalCells = MODES.size() * agentTypes.size() * orderings.size() * tasks.size()
                * STEP_BUDGETS.length * ALPHAS.length;
        System.out.println("Cost-aware induced experiment: " + MODES.size() + " modes x "
                + agentTypes.size() + " agents x " + orderings.size() + " orderings x "
                + tasks.size() + " tasks x " + STEP_BUDGETS.length + " step budgets x "
                + ALPHAS.length + " alphas x " + SEEDS_PER_CONDITION + " seeds = "
                + totalCells + " cells, " + PARALLELISM + " threads.");

        Path outputDir = Path.of("output");
        Files.createDirectories(outputDir);

        List<Callable<CellResult>> cellTasks = new ArrayList<>();
        for (GovernorMode mode : MODES) {
            for (AgentType agentType : agentTypes) {
                for (Ordering ordering : orderings) {
                    for (MockTaskSpec task : tasks) {
                        for (int maxSteps : STEP_BUDGETS) {
                            for (double alpha : ALPHAS) {
                                cellTasks.add(buildCellTask(mode, agentType, ordering, task, maxSteps, alpha));
                            }
                        }
                    }
                }
            }
        }

        ExecutorService executor = Executors.newFixedThreadPool(PARALLELISM);
        List<RunRecord> allRecords = new ArrayList<>();
        List<String> runRows = new ArrayList<>();
        runRows.add("seed,task_id,agent_type,mode,ordering,lag_steps,alpha,max_steps,"
                + AgentRunMetrics.emptyHeaderWithoutAgentPrefix()
                + ",soft_warnings,hard_blocks,mean_entropy,mean_regulation_intensity,"
                + "induced_failures,backoff_steps");

        try {
            CompletionService<CellResult> completionService = new ExecutorCompletionService<>(executor);
            for (Callable<CellResult> cellTask : cellTasks) {
                completionService.submit(cellTask);
            }
            for (int done = 1; done <= cellTasks.size(); done++) {
                CellResult result = completionService.take().get();
                allRecords.addAll(result.records());
                runRows.addAll(result.runRows());
                if (done % 40 == 0 || done == cellTasks.size()) {
                    System.out.println("  " + done + "/" + cellTasks.size() + " cells complete");
                }
            }
        } finally {
            executor.shutdown();
        }

        deleteDirectoryIfExists(WORKSPACE_ROOT);

        Path runsCsv = outputDir.resolve("cost_aware_runs.csv");
        Files.write(runsCsv, runRows, StandardCharsets.UTF_8);
        System.out.println("Wrote: " + runsCsv.toAbsolutePath());

        List<String> aggregateRows = buildAggregateRows(allRecords);
        Path aggregateCsv = outputDir.resolve("cost_aware_aggregate.csv");
        Files.write(aggregateCsv, aggregateRows, StandardCharsets.UTF_8);
        System.out.println("Wrote: " + aggregateCsv.toAbsolutePath());
    }

    private static Callable<CellResult> buildCellTask(
            GovernorMode mode, AgentType agentType, Ordering ordering,
            MockTaskSpec task, int maxSteps, double alpha
    ) {
        return () -> {
            List<RunRecord> localRecords = new ArrayList<>();
            List<String> localRunRows = new ArrayList<>();

            String cellDirName = mode.name() + "_" + agentType.name() + "_" + ordering.name() + "_"
                    + task.taskId() + "_" + maxSteps + "_a" + fmtAlpha(alpha);
            Path cellWorkspace = WORKSPACE_ROOT.resolve(cellDirName);

            try {
                for (int seed = 0; seed < SEEDS_PER_CONDITION; seed++) {
                    runOne(localRecords, localRunRows, seed, task, agentType, ordering, mode, maxSteps, alpha, cellWorkspace);
                }
            } finally {
                deleteDirectoryIfExists(cellWorkspace);
            }
            return new CellResult(localRecords, localRunRows);
        };
    }

    private static void runOne(
            List<RunRecord> records, List<String> runRows, int seed, MockTaskSpec task,
            AgentType agentType, Ordering ordering, GovernorMode mode, int maxSteps, double alpha, Path workspace
    ) throws IOException {
        resetWorkspace(task, workspace);

        ToolRegistry tools = new ToolRegistry();
        tools.register(new FileTool(workspace.toString()));

        // Mode is deliberately excluded from the seed so every mode shares
        // streams per (seed, task, ordering, alpha) as far as their differing
        // random draws allow.
        long policySeed = seedFor(seed, task.taskId(), agentType.name(), ordering.name(), maxSteps, alpha, "policy");
        long inducedSeed = seedFor(seed, task.taskId(), agentType.name(), ordering.name(), maxSteps, alpha, "induced");

        ActionPolicy policy = new ActionPolicy(
                agentType.parameters(), new Random(policySeed),
                task.expectedTarget(), task.expectedAnswer());

        CASGovernor governor = CASGovernor.forExpectedTarget(task.expectedTarget());

        InducedCostStochasticToolAgent agent = new InducedCostStochasticToolAgent(
                policy, tools, governor, true, true, ordering.lagSteps(), alpha, new Random(inducedSeed),
                mode.costAware(), mode.lambda(), mode.k());

        agent.run(task.prompt(), maxSteps);

        String metricName = mode.name() + "_" + agentType.name() + "_" + ordering.name()
                + "_a" + fmtAlpha(alpha) + "_steps" + maxSteps;
        AgentRunMetrics metrics = AgentRunMetrics.fromHistory(metricName, agent.history(), task.expectedAnswer());

        int softWarnings = countMessages(agent.history(), "SOFT_WARNING:");
        int hardBlocks = countMessages(agent.history(), "HARD_BLOCK:");
        double meanEntropy = mean(agent.entropyHistory());
        double meanIntensity = mean(agent.regulationIntensityHistory());
        int inducedFailures = agent.inducedFailureCount();
        int backoffSteps = agent.backoffStepCount();

        records.add(new RunRecord(
                agentType.name(), mode.name(), ordering.name(), task.taskId(), maxSteps, alpha,
                metrics, hardBlocks, meanIntensity, inducedFailures, backoffSteps));

        runRows.add(String.join(",",
                Integer.toString(seed),
                task.taskId(),
                agentType.name(),
                mode.name(),
                ordering.name(),
                Integer.toString(ordering.lagSteps()),
                fmtAlpha(alpha),
                Integer.toString(maxSteps),
                metrics.toCsvRow(),
                Integer.toString(softWarnings),
                Integer.toString(hardBlocks),
                fmt(meanEntropy),
                fmt(meanIntensity),
                Integer.toString(inducedFailures),
                Integer.toString(backoffSteps)));
    }

    private static List<String> buildAggregateRows(List<RunRecord> records) {
        List<String> rows = new ArrayList<>();
        rows.add("agent_type,mode,ordering,max_steps,alpha,n,success_rate,answered_rate,"
                + "mean_hard_blocks,mean_post_error_tool_burden,mean_regulation_intensity,"
                + "mean_induced_failures,mean_backoff_steps,sd_post_error_tool_burden");

        record CellKey(String agentType, String mode, String ordering, int maxSteps, double alpha) {
        }
        java.util.Map<CellKey, List<RunRecord>> grouped = new java.util.LinkedHashMap<>();
        for (RunRecord r : records) {
            grouped.computeIfAbsent(new CellKey(r.agentType(), r.mode(), r.ordering(), r.maxSteps(), r.alpha()),
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
            double meanBackoff = cell.stream().mapToInt(RunRecord::backoffSteps).average().orElse(0.0);
            double sdBurden = sampleSd(cell.stream().mapToDouble(r -> r.metrics().postErrorToolBurden()).toArray());

            rows.add(String.join(",",
                    entry.getKey().agentType(),
                    entry.getKey().mode(),
                    entry.getKey().ordering(),
                    Integer.toString(entry.getKey().maxSteps()),
                    fmtAlpha(entry.getKey().alpha()),
                    Integer.toString(cell.size()),
                    fmt(successRate), fmt(answeredRate), fmt(meanHardBlocks),
                    fmt(meanBurden), fmt(meanIntensity), fmt(meanInduced), fmt(meanBackoff), fmt(sdBurden)));
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
        return String.format("%.2f", alpha);
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
            String agentType, String mode, String ordering, String taskId, int maxSteps, double alpha,
            AgentRunMetrics metrics, int hardBlocks, double meanRegulationIntensity,
            int inducedFailures, int backoffSteps
    ) {
    }

    private record CellResult(List<RunRecord> records, List<String> runRows) {
    }
}
