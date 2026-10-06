package org.cas.agentgovernor.experiment;

import org.cas.agentgovernor.agents.GovernedLLMAgent;
import org.cas.agentgovernor.core.CASGovernor;
import org.cas.agentgovernor.llm.GeminiLLMClient;
import org.cas.agentgovernor.llm.MockClingingLLM;
import org.cas.agentgovernor.metrics.AgentRunMetrics;
import org.cas.agentgovernor.tools.FileTool;
import org.cas.agentgovernor.tools.ToolRegistry;

public class RunGeminiDemo {

    public static void main(String[] args) throws Exception {

        ToolRegistry tools = new ToolRegistry();
        tools.register(new FileTool("test_workspace"));

        CASGovernor governor = new CASGovernor();
        // No fixed oracle answer here: this demo drives a live LLM with a free-text
        // task, so there's no ground-truth string to compare against. success()/correct()
        // will therefore always read false; use answered() to see whether it stopped
        // and produced any final answer at all, and read the printed answer by eye.

        String task = """
                Find the trial config file and summarize the baseline,
                disturbance, and recovery phases.

                The config is probably examples/trial_config.json.
                """;

        String answer;
        AgentRunMetrics metrics;

        try (GeminiLLMClient llm = new GeminiLLMClient("gemini-2.5-flash")) {
            GovernedLLMAgent agent = new GovernedLLMAgent(
                llm,
                governor,
                tools,
                30000L
            );

            answer = agent.run(task, 8);

            System.out.println("\n=== FINAL ANSWER ===");
            System.out.println(answer);

            System.out.println("\n=== HISTORY ===");
            agent.history().forEach(System.out::println);

            metrics = AgentRunMetrics.fromHistory("governed_mock", agent.history(), null);
        }

        System.out.println("\n=== METRICS ===");
        System.out.println(metrics.toCsvHeader());
        System.out.println(metrics.toCsvRow());
        System.out.println(metrics);
    }
    
}
