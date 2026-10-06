package org.cas.agentgovernor.experiment;

import org.cas.agentgovernor.agents.GovernedLLMAgent;
import org.cas.agentgovernor.core.CASGovernor;
import org.cas.agentgovernor.llm.MockClingingLLM;
import org.cas.agentgovernor.metrics.AgentRunMetrics;
import org.cas.agentgovernor.tools.FileTool;
import org.cas.agentgovernor.tools.ToolRegistry;

public class RunMockDemo {

    public static void main(String[] args) throws Exception {

        ToolRegistry tools = new ToolRegistry();
        tools.register(new FileTool("test_workspace"));

        MockClingingLLM llm = new MockClingingLLM();
        CASGovernor governor = CASGovernor.forExpectedTarget(llm.expectedTarget());
        GovernedLLMAgent agent = new GovernedLLMAgent(
            llm,
            governor,
            tools
        );

        String task = """
                Find the trial config file and summarize the baseline,
                disturbance, and recovery phases.

                The config is probably examples/trial_config.json.
                """;

        String answer = agent.run(task, 8);

        System.out.println("\n=== FINAL ANSWER ===");
        System.out.println(answer);

        System.out.println("\n=== HISTORY ===");
        agent.history().forEach(System.out::println);
        
        AgentRunMetrics metrics =
        AgentRunMetrics.fromHistory("governed_mock", agent.history(), llm.expectedAnswer());

        System.out.println("\n=== METRICS ===");
        System.out.println(metrics.toCsvHeader());
        System.out.println(metrics.toCsvRow());
        System.out.println(metrics);
    }
    
}
