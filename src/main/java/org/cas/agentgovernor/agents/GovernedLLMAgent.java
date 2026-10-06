/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package org.cas.agentgovernor.agents;

import java.util.ArrayList;
import java.util.List;

import org.cas.agentgovernor.core.AgentEvent;
import org.cas.agentgovernor.core.CASGovernor;
import org.cas.agentgovernor.core.GovernorDecision;
import org.cas.agentgovernor.core.RegulatorySignal;
import org.cas.agentgovernor.core.ToolAction;
import org.cas.agentgovernor.core.ToolResult;
import org.cas.agentgovernor.llm.LLMClient;
import org.cas.agentgovernor.llm.LLMRequest;
import org.cas.agentgovernor.llm.LLMResponse;
import org.cas.agentgovernor.tools.ToolRegistry;


/**
 * Real-LLM-backed governed agent. The regulatory signal computed each step
 * both (a) gets embedded in the prompt sent to the LLM via {@code LLMRequest},
 * and (b) drives the governor's block/warn decisions on the LLM's proposed
 * action or final answer.
 *
 * <p>Supports an optional sham/placebo mode (see the 5-arg constructor):
 * when {@code scriptedControlSignals} is supplied, the signal used for BOTH
 * of those roles at step {@code i} is {@code scriptedControlSignals.get(i)}
 * rather than the real, live {@code governor.evaluate(history)} result --
 * i.e. the LLM sees, and the governor blocks based on, a state that is NOT
 * contingent on what has actually happened in this episode. This is the
 * real-LLM analogue of {@code StochasticToolAgent}'s sham mode: pair it with
 * a shuffled copy of a real governed run's realized signal sequence (see
 * {@code RunGeminiTaskComparison}) to test whether a governed agent's
 * apparent advantage is attributable to the signal actually tracking real
 * disturbance history, or would hold even under a decorrelated but
 * equally-frequent, equally-strong signal.
 *
 * @author veronique
 */

public class GovernedLLMAgent {

    private final LLMClient llm;
    private final CASGovernor governor;
    private final ToolRegistry tools;
    private final List<AgentEvent> history = new ArrayList<>();
    private final long delayMillis;

    /** If non-null, activates sham mode -- see class doc. */
    private final List<RegulatorySignal> scriptedControlSignals;

    /** The control signal actually used at each step, in order -- see class doc. */
    private final List<RegulatorySignal> controlSignalHistory = new ArrayList<>();

    public GovernedLLMAgent(
            LLMClient llm,
            CASGovernor governor,
            ToolRegistry tools
    ) {
        this(llm, governor, tools, 0L, null);
    }

    public GovernedLLMAgent(
            LLMClient llm,
            CASGovernor governor,
            ToolRegistry tools,
            long delayMillis
    ) {
        this(llm, governor, tools, delayMillis, null);
    }

    /**
     * @param scriptedControlSignals see class doc. Pass {@code null} for
     *                               normal (non-sham) operation.
     */
    public GovernedLLMAgent(
            LLMClient llm,
            CASGovernor governor,
            ToolRegistry tools,
            long delayMillis,
            List<RegulatorySignal> scriptedControlSignals
    ) {
        this.llm = llm;
        this.governor = governor;
        this.tools = tools;
        this.delayMillis = delayMillis;
        this.scriptedControlSignals = scriptedControlSignals;
    }

    public String run(String userTask, int maxSteps) throws Exception {
        boolean sham = scriptedControlSignals != null;

        for (int step = 0; step < maxSteps; step++) {

            // Observational signal: ALWAYS computed from the real, live
            // history and recorded, regardless of mode -- consistent with
            // StochasticToolAgent's rationale, this is what any future
            // metrics/analysis of regulatory-state counts should read,
            // kept comparable across governed and sham runs alike.
            RegulatorySignal observedSignal = governor.evaluate(history);
            history.add(AgentEvent.regulatorySignal(observedSignal));

            RegulatorySignal controlSignal;
            if (sham) {
                if (step >= scriptedControlSignals.size()) {
                    // Scripted sequence exhausted before the agent finished --
                    // end the run the same way running out of maxSteps would.
                    return "Stopped: maximum number of steps reached.";
                }
                controlSignal = scriptedControlSignals.get(step);
            } else {
                controlSignal = observedSignal;
            }
            controlSignalHistory.add(controlSignal);

            LLMRequest request = new LLMRequest(userTask, history, controlSignal);

            if (step > 0 && delayMillis > 0) {
                Thread.sleep(delayMillis);
            }

            LLMResponse response = llm.generate(request);

            if (response.wantsFinalAnswer()) {
                GovernorDecision finalDecision =
                        governor.reviewFinalAnswer(controlSignal, history, response.finalAnswer());

                if (finalDecision.allowed()) {
                    history.add(AgentEvent.finalAnswer(response.finalAnswer()));
                    return response.finalAnswer();
                }

                history.add(AgentEvent.governorIntervention(
                        "HARD_BLOCK: " + finalDecision.reason()
                ));
                continue;
            }

            ToolAction proposed = response.proposedAction();

            if (proposed == null) {
                history.add(AgentEvent.governorIntervention(
                        "HARD_BLOCK: LLM returned no action and no final answer."
                ));
                continue;
            }

            GovernorDecision decision = governor.reviewAction(controlSignal, history, proposed);

            if (decision.warning()) {
                history.add(AgentEvent.governorIntervention(
                        "SOFT_WARNING: " + decision.reason()
                ));
            }

            if (decision.blocked()) {
                history.add(AgentEvent.governorIntervention(
                        "HARD_BLOCK: " + decision.reason()
                ));
                continue;
            }

            ToolResult result = tools.execute(proposed);

            history.add(AgentEvent.toolCall(proposed, result));
            governor.observe(proposed, result);
        }

        return "Stopped: maximum number of steps reached.";
    }

    public List<AgentEvent> history() {
        return history;
    }

    /** The control signal actually used at each step, in order (see class doc). */
    public List<RegulatorySignal> controlSignalHistory() {
        return controlSignalHistory;
    }
}
