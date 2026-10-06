/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package org.cas.agentgovernor.experiment;

import org.cas.agentgovernor.agents.UngovernedLLMAgent;
import org.cas.agentgovernor.llm.LLMClient;
import org.cas.agentgovernor.llm.MockClingingLLM;
import org.cas.agentgovernor.metrics.AgentRunMetrics;
import org.cas.agentgovernor.tools.FileTool;
import org.cas.agentgovernor.tools.ToolRegistry;

/**
 *
 * @author veronique
 */

public class RunMockUngovernedDemo {

    public static void main(String[] args) throws Exception {

        ToolRegistry tools = new ToolRegistry();
        tools.register(new FileTool("test_workspace"));

        LLMClient llm = new MockClingingLLM();

        UngovernedLLMAgent agent = new UngovernedLLMAgent(
                llm,
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
        
        AgentRunMetrics metrics = AgentRunMetrics.fromHistory(
                "ungoverned_mock", agent.history(), ((MockClingingLLM) llm).expectedAnswer());

        System.out.println("\n=== METRICS ===");
        System.out.println(metrics.toCsvHeader());
        System.out.println(metrics.toCsvRow());
        System.out.println(metrics);
    }
}
