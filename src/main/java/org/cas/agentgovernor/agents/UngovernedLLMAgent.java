/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package org.cas.agentgovernor.agents;

import java.util.ArrayList;
import java.util.List;

import org.cas.agentgovernor.core.AgentEvent;
import org.cas.agentgovernor.core.ToolAction;
import org.cas.agentgovernor.core.ToolResult;
import org.cas.agentgovernor.llm.LLMClient;
import org.cas.agentgovernor.llm.LLMRequest;
import org.cas.agentgovernor.llm.LLMResponse;
import org.cas.agentgovernor.tools.ToolRegistry;

/**
 *
 * @author veronique
 */

public class UngovernedLLMAgent {

    private final LLMClient llm;
    private final ToolRegistry tools;
    private final List<AgentEvent> history = new ArrayList<>();

    private final long delayMillis;

    public UngovernedLLMAgent(
            LLMClient llm,
            ToolRegistry tools
    ) {
        this(llm, tools, 0L);
    }

    public UngovernedLLMAgent(
            LLMClient llm,
            ToolRegistry tools,
            long delayMillis
    ) {
        this.llm = llm;
        this.tools = tools;
        this.delayMillis = delayMillis;
    }
    
    public String run(String userTask, int maxSteps) throws Exception {
        for (int step = 0; step < maxSteps; step++) {

            LLMRequest request = new LLMRequest(userTask, history);
            LLMResponse response = llm.generate(request);

            if (response.wantsFinalAnswer()) {
                history.add(AgentEvent.finalAnswer(response.finalAnswer()));
                return response.finalAnswer();
            }

            ToolAction proposed = response.proposedAction();

            if (proposed == null) {
                history.add(AgentEvent.note("LLM returned no action and no final answer."));
                continue;
            }

            ToolResult result = tools.execute(proposed);
            history.add(AgentEvent.toolCall(proposed, result));
        }

        return "Stopped: maximum number of steps reached.";
    }

    public List<AgentEvent> history() {
        return history;
    }
}
