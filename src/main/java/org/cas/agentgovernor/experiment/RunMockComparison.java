package org.cas.agentgovernor.experiment;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import org.cas.agentgovernor.agents.GovernedLLMAgent;
import org.cas.agentgovernor.agents.UngovernedLLMAgent;
import org.cas.agentgovernor.core.CASGovernor;
import org.cas.agentgovernor.experiment.tasks.MockTaskSet;
import org.cas.agentgovernor.experiment.tasks.MockTaskSpec;
import org.cas.agentgovernor.llm.LLMClient;
import org.cas.agentgovernor.llm.MockClingingLLM;
import org.cas.agentgovernor.llm.MockPersistentLLM;
import org.cas.agentgovernor.metrics.AgentRunMetrics;
import org.cas.agentgovernor.tools.FileTool;
import org.cas.agentgovernor.tools.ToolRegistry;

public class RunMockComparison {

    private static final Path WORKSPACE = Path.of("test_workspace");
    private static final Path OUTPUT_DIR = Path.of("output");

    public static void main(String[] args) throws Exception {
        Files.createDirectories(OUTPUT_DIR);

        List<ComparisonRecord> records = new ArrayList<>();
        List<String> csvRows = new ArrayList<>();

        String header = "mock_type,task_id," + AgentRunMetrics.emptyHeaderWithoutAgentPrefix();
        csvRows.add(header);

        runSuite("clinging", records, csvRows);
        runSuite("persistent", records, csvRows);

        Path summaryCsv = OUTPUT_DIR.resolve("mock_task_comparison_summary.csv");
        Files.writeString(
                summaryCsv,
                String.join(System.lineSeparator(), csvRows) + System.lineSeparator(),
                StandardCharsets.UTF_8
        );

        List<String> aggregateRows = buildAggregateRows(records);

        Path aggregateCsv = OUTPUT_DIR.resolve("mock_task_aggregate_summary.csv");
        Files.writeString(
                aggregateCsv,
                String.join(System.lineSeparator(), aggregateRows) + System.lineSeparator(),
                StandardCharsets.UTF_8
        );

        printAggregateSummary(aggregateRows);

        System.out.println("\n========================================");
        System.out.println("WROTE CSV FILES");
        System.out.println("========================================");
        System.out.println(summaryCsv.toAbsolutePath());
        System.out.println(aggregateCsv.toAbsolutePath());
    }

    private static void runSuite(
            String mockType,
            List<ComparisonRecord> records,
            List<String> csvRows
    ) throws Exception {

        for (MockTaskSpec task : MockTaskSet.fileRecoveryTasks()) {
            System.out.println("\n========================================");
            System.out.println("MOCK " + mockType + " | TASK " + task.taskId());
            System.out.println("========================================");

            resetWorkspace(task);
            AgentRunMetrics ungoverned = runUngoverned(mockType, task);
            records.add(new ComparisonRecord(mockType, task.taskId(), "ungoverned", ungoverned));
            csvRows.add(mockType + "," + task.taskId() + "," + ungoverned.toCsvRow());

            resetWorkspace(task);
            AgentRunMetrics governed = runGoverned(mockType, task);
            records.add(new ComparisonRecord(mockType, task.taskId(), "governed", governed));
            csvRows.add(mockType + "," + task.taskId() + "," + governed.toCsvRow());

            System.out.println("\n--- TASK COMPARISON ---");
            System.out.println("mock_type,task_id," + ungoverned.toCsvHeader());
            System.out.println(mockType + "," + task.taskId() + "," + ungoverned.toCsvRow());
            System.out.println(mockType + "," + task.taskId() + "," + governed.toCsvRow());
        }
    }

    private static AgentRunMetrics runUngoverned(String mockType, MockTaskSpec task) throws Exception {
        ToolRegistry tools = new ToolRegistry();
        tools.register(new FileTool(WORKSPACE.toString()));

        LLMClient llm = makeMockLLM(mockType, task);

        UngovernedLLMAgent agent = new UngovernedLLMAgent(llm, tools);

        String answer = agent.run(task.prompt(), 24);

        System.out.println("\n=== UNGOVERNED FINAL ANSWER ===");
        System.out.println(answer);

        System.out.println("\n=== UNGOVERNED HISTORY ===");
        agent.history().forEach(System.out::println);

        return AgentRunMetrics.fromHistory("ungoverned_mock", agent.history(), task.expectedAnswer());
    }

    private static AgentRunMetrics runGoverned(String mockType, MockTaskSpec task) throws Exception {
        ToolRegistry tools = new ToolRegistry();
        tools.register(new FileTool(WORKSPACE.toString()));

        LLMClient llm = makeMockLLM(mockType, task);
        CASGovernor governor = CASGovernor.forExpectedTarget(task.expectedTarget());

        GovernedLLMAgent agent = new GovernedLLMAgent(llm, governor, tools);

        String answer = agent.run(task.prompt(), 24);

        System.out.println("\n=== GOVERNED FINAL ANSWER ===");
        System.out.println(answer);

        System.out.println("\n=== GOVERNED HISTORY ===");
        agent.history().forEach(System.out::println);

        return AgentRunMetrics.fromHistory("governed_mock", agent.history(), task.expectedAnswer());
    }

    private static LLMClient makeMockLLM(String mockType, MockTaskSpec task) {
        if ("persistent".equalsIgnoreCase(mockType)) {
            return new MockPersistentLLM(
                    task.expectedTarget(),
                    task.expectedAnswer(),
                    4
            );
        }

        return new MockClingingLLM(
                task.expectedTarget(),
                task.expectedAnswer()
        );
    }

    private static List<String> buildAggregateRows(List<ComparisonRecord> records) {
        List<String> rows = new ArrayList<>();

        rows.add("mock_type,agent,n,success_rate,"
                + "mean_tool_calls,mean_failed_tool_calls,"
                + "mean_repeated_failed_actions,mean_repeated_successful_actions,"
                + "mean_governor_interventions,mean_regulatory_signals,"
                + "mean_disturbed_signals,mean_recovering_signals,"
                + "mean_released_signals,mean_post_error_tool_burden");

        addAggregate(rows, records, "clinging", "ungoverned");
        addAggregate(rows, records, "clinging", "governed");
        addAggregate(rows, records, "persistent", "ungoverned");
        addAggregate(rows, records, "persistent", "governed");

        return rows;
    }

    private static void addAggregate(
            List<String> rows,
            List<ComparisonRecord> records,
            String mockType,
            String condition
    ) {
        List<AgentRunMetrics> selected = new ArrayList<>();

        for (ComparisonRecord record : records) {
            if (record.mockType().equals(mockType)
                    && record.condition().equals(condition)) {
                selected.add(record.metrics());
            }
        }

        if (selected.isEmpty()) {
            return;
        }

        int n = selected.size();

        double successRate = meanSuccess(selected);
        double meanToolCalls = mean(selected, MetricField.TOTAL_TOOL_CALLS);
        double meanFailed = mean(selected, MetricField.FAILED_TOOL_CALLS);
        double meanRepeatedFailed = mean(selected, MetricField.REPEATED_FAILED_ACTIONS);
        double meanRepeatedSuccessful = mean(selected, MetricField.REPEATED_SUCCESSFUL_ACTIONS);
        double meanGovernorInterventions = mean(selected, MetricField.GOVERNOR_INTERVENTIONS);
        double meanRegulatorySignals = mean(selected, MetricField.REGULATORY_SIGNALS);
        double meanDisturbedSignals = mean(selected, MetricField.DISTURBED_SIGNALS);
        double meanRecoveringSignals = mean(selected, MetricField.RECOVERING_SIGNALS);
        double meanReleasedSignals = mean(selected, MetricField.RELEASED_SIGNALS);
        double meanPostErrorBurden = mean(selected, MetricField.POST_ERROR_TOOL_BURDEN);

        rows.add(String.join(",",
                mockType,
                condition,
                Integer.toString(n),
                fmt(successRate),
                fmt(meanToolCalls),
                fmt(meanFailed),
                fmt(meanRepeatedFailed),
                fmt(meanRepeatedSuccessful),
                fmt(meanGovernorInterventions),
                fmt(meanRegulatorySignals),
                fmt(meanDisturbedSignals),
                fmt(meanRecoveringSignals),
                fmt(meanReleasedSignals),
                fmt(meanPostErrorBurden)
        ));
    }

    private static void printAggregateSummary(List<String> aggregateRows) {
        System.out.println("\n========================================");
        System.out.println("AGGREGATE SUMMARY");
        System.out.println("========================================");

        for (String row : aggregateRows) {
            System.out.println(row);
        }
    }

    private static double meanSuccess(List<AgentRunMetrics> metrics) {
        int count = 0;

        for (AgentRunMetrics metric : metrics) {
            if (metric.success()) {
                count++;
            }
        }

        return ((double) count) / metrics.size();
    }

    private static double mean(List<AgentRunMetrics> metrics, MetricField field) {
        double sum = 0.0;

        for (AgentRunMetrics metric : metrics) {
            sum += switch (field) {
                case TOTAL_TOOL_CALLS -> metric.totalToolCalls();
                case FAILED_TOOL_CALLS -> metric.failedToolCalls();
                case REPEATED_FAILED_ACTIONS -> metric.repeatedFailedActions();
                case REPEATED_SUCCESSFUL_ACTIONS -> metric.repeatedSuccessfulActions();
                case GOVERNOR_INTERVENTIONS -> metric.governorInterventions();
                case REGULATORY_SIGNALS -> metric.regulatorySignals();
                case DISTURBED_SIGNALS -> metric.disturbedSignals();
                case RECOVERING_SIGNALS -> metric.recoveringSignals();
                case RELEASED_SIGNALS -> metric.releasedSignals();
                case POST_ERROR_TOOL_BURDEN -> metric.postErrorToolBurden();
            };
        }

        return sum / metrics.size();
    }

    private static String fmt(double value) {
        return String.format("%.3f", value);
    }

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

    private record ComparisonRecord(
            String mockType,
            String taskId,
            String condition,
            AgentRunMetrics metrics
    ) {
    }

    private enum MetricField {
        TOTAL_TOOL_CALLS,
        FAILED_TOOL_CALLS,
        REPEATED_FAILED_ACTIONS,
        REPEATED_SUCCESSFUL_ACTIONS,
        GOVERNOR_INTERVENTIONS,
        REGULATORY_SIGNALS,
        DISTURBED_SIGNALS,
        RECOVERING_SIGNALS,
        RELEASED_SIGNALS,
        POST_ERROR_TOOL_BURDEN
    }
}