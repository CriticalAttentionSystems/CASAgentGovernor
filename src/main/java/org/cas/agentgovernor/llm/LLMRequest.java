/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package org.cas.agentgovernor.llm;

import java.util.Collections;
import java.util.List;
import org.cas.agentgovernor.core.AgentEvent;
import org.cas.agentgovernor.core.RegulatorySignal;
import org.cas.agentgovernor.core.RegulatoryState;

/**
 *
 * @author veronique
 */

public class LLMRequest {

    private final String userTask;
    private final List<AgentEvent> history;
    private final RegulatorySignal regulatorySignal;

    public LLMRequest(String userTask, List<AgentEvent> history) {
        this(
                userTask,
                history,
                new RegulatorySignal(
                        RegulatoryState.NORMAL,
                        0,
                        "Normal operation."
                )
        );
    }

    public LLMRequest(
            String userTask,
            List<AgentEvent> history,
            RegulatorySignal regulatorySignal
    ) {
        this.userTask = userTask;
        this.history = history == null ? Collections.emptyList() : history;
        this.regulatorySignal = regulatorySignal;
    }

    public String userTask() {
        return userTask;
    }

    public List<AgentEvent> history() {
        return history;
    }

    public RegulatorySignal regulatorySignal() {
        return regulatorySignal;
    }
}
