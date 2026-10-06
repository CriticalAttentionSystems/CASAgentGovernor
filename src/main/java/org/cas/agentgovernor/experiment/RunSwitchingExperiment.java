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

import org.cas.agentgovernor.agents.SwitchingStochasticToolAgent;
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
 * Tests whether "Intermittent Control Is Not Diluted Control" (paper 3's
 * switching-relief finding) generalizes to the discrete tool-use
 * substrate this whole project is built on. Paper 3's own result: an
 * agent alternating between RF and DF causal ordering requires LESS mean
 * regulatory gain than an occupancy-weighted mixture of fixed-RF and
 * fixed-DF trajectories would predict -- switching is not diluted
 * benefit, it is a genuine, nonlinear relief effect, confirmed at
 * N=1000/200,000 steps in the original continuous-state simulation.
 *
 * <p>This class produces ONLY the switching trajectories, at three dwell
 * lengths (L=1, 2, 3 steps) -- kept small deliberately: at these 16/32-step
 * episode lengths, longer dwell blocks let many episodes finish entirely
 * within their first RF block before ever reaching a DF block, which
 * skews realized RF occupancy well above the intended 50/50 and makes
 * dwell-length comparisons unreliable. An earlier version of this class
 * used L=1,4,8 and confirmed exactly this: mean_p_rf drifted from 0.55 at
 * L=1 to 0.89 at L=8. Kept small, the schedule actually gets to alternate
 * within a typical episode's length. That scale mismatch with paper 3
 * (200,000 steps, dwell lengths up to 2000) is real regardless and should
 * be stated plainly when interpreting results, not hidden. The
 * occupancy-weighted reference is NOT computed here -- it is built in the
 * accompanying analysis script from the
 * ALREADY-COLLECTED governed (RF, lag=0) and disturbance_first (DF,
 * lag=1) data in stochastic_policy_aggregate.csv, matching paper 3's own
 * methodology of comparing against fixed-mode baselines from the same
 * underlying model.
 *
 * <p>DF dwell blocks use regulationLagSteps=1, exactly matching the
 * existing disturbance_first condition's lag value, so the fixed-mode
 * reference this gets compared against is a genuine match, not an
 * approximation.
 *
 * @author veronique
 */
public class RunSwitchingExperiment {

    private static final Path WORKSPACE_ROOT = Path.of("test_workspace_switching");
    private static final int SEEDS_PER_CONDITION = 2000;
    private static final int DF_LAG_STEPS = 1; // matches disturbance_first exactly
    private static final int[] DWELL_LENGTHS = {1, 2, 3};
    private static final int[] STEP_BUDGETS = {16, 32}; // need room for multiple dwell cycles
    private static final int PARALLELISM = Math.max(1, Runtime.getRuntime().availableProcessors());

    private record AgentType(String name, PolicyParameters parameters) {
    }

    public static void main(String[] args) throws Exception {
        List<AgentType> agentTypes = List.of(
                new AgentType("exploratory", PolicyParameters.exploratory()),
                new AgentType("persistent", PolicyParameters.persistent())
        );

        List<MockTaskSpec> tasks = MockTaskSet.fileRecoveryTasks();

        System.out.println("Switching experiment: " + agentTypes.size() + " agent types x " + tasks.size()
                + " tasks x " + STEP_BUDGETS.length + " step budgets x " + DWELL_LENGTHS.length
                + " dwell lengths x " + SEEDS_PER_CONDITION + " seeds, " + PARALLELISM + " threads.");

        Path outputDir = Path.of("output");
        Files.createDirectories(outputDir);

        List<Callable<CellResult>> cellTasks = new ArrayList<>();
        for (AgentType agentType : agentTypes) {
            for (MockTaskSpec task : tasks) {
                for (int maxSteps : STEP_BUDGETS) {
                    for (int dwellLength : DWELL_LENGTHS) {
                        cellTasks.add(buildCellTask(agentType, task, maxSteps, dwellLength));
                    }
                }
            }
        }

        int totalCells = cellTasks.size();
        ExecutorService executor = Executors.newFixedThreadPool(PARALLELISM);

        List<RunRecord> allRecords = new ArrayList<>();
        List<String> runRows = new ArrayList<>();
        runRows.add("seed,task_id,agent_type,dwell_length,max_steps,"
                + AgentRunMetrics.emptyHeaderWithoutAgentPrefix()
                + ",soft_warnings,hard_blocks,mean_entropy,rf_steps,df_steps");

        try {
            // CompletionService, not invokeAll: invokeAll blocks until the
            // last cell finishes, which made the progress counter below
            // print every line at once at the very end.
            CompletionService<CellResult> completionService =
                    new ExecutorCompletionService<>(executor);
            for (Callable<CellResult> cellTask : cellTasks) {
                completionService.submit(cellTask);
            }

            for (int done = 1; done <= totalCells; done++) {
                CellResult result = completionService.take().get();
                allRecords.addAll(result.records());
                runRows.addAll(result.runRows());

                if (done % 5 == 0 || done == totalCells) {
                    System.out.println("  " + done + "/" + totalCells + " cells complete");
                }
            }
        } finally {
            executor.shutdown();
        }

        deleteDirectoryIfExists(WORKSPACE_ROOT);

        Path runsCsv = outputDir.resolve("switching_runs.csv");
        Files.write(runsCsv, runRows, StandardCharsets.UTF_8);
        System.out.println("Wrote: " + runsCsv.toAbsolutePath());

        List<String> aggregateRows = buildAggregateRows(allRecords);
        Path aggregateCsv = outputDir.resolve("switching_aggregate.csv");
        Files.write(aggregateCsv, aggregateRows, StandardCharsets.UTF_8);
        System.out.println("Wrote: " + aggregateCsv.toAbsolutePath());
    }

    private static Callable<CellResult> buildCellTask(
            AgentType agentType, MockTaskSpec task, int maxSteps, int dwellLength
    ) {
        return () -> {
            List<RunRecord> localRecords = new ArrayList<>();
            List<String> localRunRows = new ArrayList<>();

            String cellDirName = agentType.name() + "_" + task.taskId() + "_" + maxSteps + "_L" + dwellLength;
            Path cellWorkspace = WORKSPACE_ROOT.resolve(cellDirName);

            try {
                for (int seed = 0; seed < SEEDS_PER_CONDITION; seed++) {
                    long seedValue = seedFor(seed, task.taskId(), agentType.name(), dwellLength, maxSteps);
                    runOne(localRecords, localRunRows, seed, task, agentType, maxSteps, dwellLength, seedValue, cellWorkspace);
                }
            } finally {
                deleteDirectoryIfExists(cellWorkspace);
            }

            return new CellResult(localRecords, localRunRows);
        };
    }

    private static void runOne(
            List<RunRecord> records,
            List<String> runRows,
            int seed,
            MockTaskSpec task,
            AgentType agentType,
            int maxSteps,
            int dwellLength,
            long seedValue,
            Path workspace
    ) throws IOException {
        resetWorkspace(task, workspace);

        ToolRegistry tools = new ToolRegistry();
        tools.register(new FileTool(workspace.toString()));

        Random random = new Random(seedValue);

        ActionPolicy policy = new ActionPolicy(
                agentType.parameters(),
                random,
                task.expectedTarget(),
                task.expectedAnswer()
        );

        CASGovernor governor = CASGovernor.forExpectedTarget(task.expectedTarget());

        SwitchingStochasticToolAgent agent = new SwitchingStochasticToolAgent(
                policy, tools, governor, true, true, dwellLength, DF_LAG_STEPS
        );

        agent.run(task.prompt(), maxSteps);

        String metricName = agentType.name() + "_switching_L" + dwellLength + "_steps" + maxSteps;
        AgentRunMetrics metrics = AgentRunMetrics.fromHistory(metricName, agent.history(), task.expectedAnswer());

        int softWarnings = countMessages(agent.history(), "SOFT_WARNING:");
        int hardBlocks = countMessages(agent.history(), "HARD_BLOCK:");
        double meanEntropy = mean(agent.entropyHistory());

        long rfSteps = agent.modeHistory().stream().filter(m -> m.equals("RF")).count();
        long dfSteps = agent.modeHistory().stream().filter(m -> m.equals("DF")).count();

        records.add(new RunRecord(
                agentType.name(), task.taskId(), dwellLength, maxSteps,
                metrics, softWarnings, hardBlocks, meanEntropy
        ));

        runRows.add(String.join(",",
                Integer.toString(seed),
                task.taskId(),
                agentType.name(),
                Integer.toString(dwellLength),
                Integer.toString(maxSteps),
                metrics.toCsvRow(),
                Integer.toString(softWarnings),
                Integer.toString(hardBlocks),
                fmt(meanEntropy),
                Long.toString(rfSteps),
                Long.toString(dfSteps)
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

    private static String fmt(double value) {
        return String.format("%.5f", value);
    }

    private static long seedFor(int seed, String taskId, String agentType, int dwellLength, int maxSteps) {
        long h = 1125899906842597L;
        h = 31 * h + seed;
        h = 31 * h + taskId.hashCode();
        h = 31 * h + agentType.hashCode();
        h = 31 * h + ("switching_L" + dwellLength).hashCode();
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

    private static List<String> buildAggregateRows(List<RunRecord> records) {
        List<String> rows = new ArrayList<>();
        rows.add("agent_type,dwell_length,max_steps,n,success_rate,answered_rate,"
                + "mean_hard_blocks,mean_post_error_tool_burden,mean_entropy");

        record CellKey(String agentType, int dwellLength, int maxSteps) {
        }

        java.util.Map<CellKey, List<RunRecord>> grouped = new java.util.LinkedHashMap<>();
        for (RunRecord r : records) {
            grouped.computeIfAbsent(new CellKey(r.agentType(), r.dwellLength(), r.maxSteps()),
                    k -> new ArrayList<>()).add(r);
        }

        for (var entry : grouped.entrySet()) {
            List<RunRecord> cell = entry.getValue();
            double successRate = cell.stream().mapToDouble(r -> r.metrics().success() ? 1.0 : 0.0).average().orElse(0.0);
            double answeredRate = cell.stream().mapToDouble(r -> r.metrics().answered() ? 1.0 : 0.0).average().orElse(0.0);
            double meanHardBlocks = cell.stream().mapToInt(RunRecord::hardBlocks).average().orElse(0.0);
            double meanBurden = cell.stream().mapToDouble(r -> r.metrics().postErrorToolBurden()).average().orElse(0.0);
            double meanEntropy = cell.stream().mapToDouble(RunRecord::meanEntropy).average().orElse(0.0);

            rows.add(String.join(",",
                    entry.getKey().agentType(),
                    Integer.toString(entry.getKey().dwellLength()),
                    Integer.toString(entry.getKey().maxSteps()),
                    Integer.toString(cell.size()),
                    fmt(successRate),
                    fmt(answeredRate),
                    fmt(meanHardBlocks),
                    fmt(meanBurden),
                    fmt(meanEntropy)
            ));
        }

        return rows;
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

        // Part of the task environment, NOT incidental scaffolding, and it
        // must match RunStochasticPolicyExperiment.resetWorkspace exactly:
        // the fixed-RF/fixed-DF runs this experiment is compared against are
        // produced there, so any difference in the workspace makes the
        // occupancy-weighted reference an invalid null. T004 in particular
        // ("empty expected directory") depends on examples/ existing and
        // being EMPTY -- omitting it makes the directory absent instead,
        // which is a different task.
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
            String agentType,
            String taskId,
            int dwellLength,
            int maxSteps,
            AgentRunMetrics metrics,
            int softWarnings,
            int hardBlocks,
            double meanEntropy
    ) {
    }

    private record CellResult(List<RunRecord> records, List<String> runRows) {
    }
}
