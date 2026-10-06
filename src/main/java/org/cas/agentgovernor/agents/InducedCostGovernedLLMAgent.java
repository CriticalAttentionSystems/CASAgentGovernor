/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package org.cas.agentgovernor.agents;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.cas.agentgovernor.core.AgentEvent;
import org.cas.agentgovernor.core.CASGovernor;
import org.cas.agentgovernor.core.GovernorDecision;
import org.cas.agentgovernor.core.RegulatorySignal;
import org.cas.agentgovernor.core.RegulatoryState;
import org.cas.agentgovernor.core.ToolAction;
import org.cas.agentgovernor.core.ToolResult;
import org.cas.agentgovernor.llm.LLMClient;
import org.cas.agentgovernor.llm.LLMRequest;
import org.cas.agentgovernor.llm.LLMResponse;
import org.cas.agentgovernor.tools.ToolRegistry;

/**
 * Real-LLM port of {@link InducedCostStochasticToolAgent}: gives the
 * governor's own tightening an induced cost while a genuine {@link LLMClient}
 * (Gemini, or a mock) drives action selection. This is the real-model version
 * of the paper-4 control-generated-disturbance test, so a result here speaks
 * to an actual LLM agent rather than a hand-coded stochastic policy.
 *
 * <h2>What tightening and induced cost mean for an LLM agent</h2>
 * The discrete governor's clamp is expressed through the regulatory STATE it
 * emits, which for an LLM agent does two things: it is embedded in the prompt
 * (a stronger state = a more insistent recovery instruction) and it drives the
 * governor's block/warn decisions. The same intensity ladder as the stochastic
 * agent applies -- {@code g(NORMAL)=0 ... g(STUCK)=0.95} -- and positive
 * tightening {@code dgPlus = max(0, g(t)-g(t-1))} generates an induced failure
 * with probability {@code min(1, alpha*dgPlus^2)}, injected as a TRANSIENT
 * failure on the next tool call. That failure enters the real history the LLM
 * sees, so the same feedback loop exists: induced failure -> governor
 * escalates -> tightens harder -> more induced failures.
 *
 * <h2>Cost-blind vs cost-aware</h2>
 * Identical to the stochastic agent. Cost-blind ({@code costAware=false}):
 * the governor always escalates and its state is always applied. Cost-aware:
 * an EWMA of the induced-failure rate lowers {@code pApply}; when backed off,
 * the strong regulatory state is NOT applied -- the agent is shown a NORMAL
 * signal and the governor does not block that step -- so intensity is 0, no
 * new induced cost is generated, and the rate decays. This is the LLM analog
 * of paper 4's cost-aware controller reducing its own regulation.
 *
 * <p>At {@code alpha==0} in cost-blind mode this is exactly a
 * {@link GovernedLLMAgent} run (same prompt, same block logic), the intended
 * regression check. Backoff is a no-op at alpha 0 (no induced failures ->
 * pApply stays 1).
 *
 * @author veronique
 */
public class InducedCostGovernedLLMAgent {

    private static final RegulatorySignal HANDS_OFF_SIGNAL = new RegulatorySignal(
            RegulatoryState.NORMAL, 0,
            "Normal operation. Proceed with the task.");

    private final LLMClient llm;
    private final CASGovernor governor;
    private final ToolRegistry tools;
    private final long delayMillis;
    private final double alpha;
    private final Random inducedRandom;
    private final boolean costAware;
    private final double backoffLambda;
    private final double backoffK;

    private final List<AgentEvent> history = new ArrayList<>();
    private final List<RegulatorySignal> controlSignalHistory = new ArrayList<>();
    private final List<Double> regulationIntensityHistory = new ArrayList<>();

    private int inducedFailureCount = 0;
    private int backoffStepCount = 0;

    public InducedCostGovernedLLMAgent(
            LLMClient llm,
            CASGovernor governor,
            ToolRegistry tools,
            long delayMillis,
            double alpha,
            Random inducedRandom,
            boolean costAware,
            double backoffLambda,
            double backoffK
    ) {
        this.llm = llm;
        this.governor = governor;
        this.tools = tools;
        this.delayMillis = delayMillis;
        this.alpha = Math.max(0.0, alpha);
        this.inducedRandom = inducedRandom;
        this.costAware = costAware;
        this.backoffLambda = backoffLambda;
        this.backoffK = backoffK;
    }

    private static double intensityOf(RegulatoryState state) {
        return switch (state) {
            case NORMAL -> 0.00;
            case DISTURBED -> 0.75;
            case RECOVERING -> 0.85;
            case STUCK -> 0.95;
            case RELEASED -> 0.99;
        };
    }

    public String run(String userTask, int maxSteps) throws Exception {
        double previousIntensity = 0.0;
        boolean induceNextToolCall = false;
        double inducedRate = 0.0;

        for (int step = 0; step < maxSteps; step++) {
            RegulatorySignal observedSignal = governor.evaluate(history);
            history.add(AgentEvent.regulatorySignal(observedSignal));

            // Cost-aware backoff: decide whether the governor applies its
            // regulation this step at all.
            boolean regulationActive = true;
            if (costAware) {
                double pApply = 1.0 - Math.min(1.0, backoffK * inducedRate);
                regulationActive = inducedRandom.nextDouble() < pApply;
                if (!regulationActive) {
                    backoffStepCount++;
                }
            }

            // The control signal the LLM is shown AND the governor blocks on.
            // When backed off, the agent is shown NORMAL and no block fires.
            RegulatorySignal controlSignal = regulationActive ? observedSignal : HANDS_OFF_SIGNAL;
            controlSignalHistory.add(controlSignal);

            double intensity = regulationActive ? intensityOf(observedSignal.state()) : 0.0;
            regulationIntensityHistory.add(intensity);
            double dgPlus = Math.max(0.0, intensity - previousIntensity);
            previousIntensity = intensity;

            if (alpha > 0.0 && dgPlus > 0.0) {
                double inducedProbability = Math.min(1.0, alpha * dgPlus * dgPlus);
                if (inducedRandom.nextDouble() < inducedProbability) {
                    induceNextToolCall = true;
                }
            }

            LLMRequest request = new LLMRequest(userTask, history, controlSignal);

            if (step > 0 && delayMillis > 0) {
                Thread.sleep(delayMillis);
            }

            LLMResponse response = llm.generate(request);

            if (response.wantsFinalAnswer()) {
                if (!regulationActive) {
                    history.add(AgentEvent.finalAnswer(response.finalAnswer()));
                    return response.finalAnswer();
                }
                GovernorDecision finalDecision =
                        governor.reviewFinalAnswer(controlSignal, history, response.finalAnswer());
                if (finalDecision.allowed()) {
                    history.add(AgentEvent.finalAnswer(response.finalAnswer()));
                    return response.finalAnswer();
                }
                history.add(AgentEvent.governorIntervention("HARD_BLOCK: " + finalDecision.reason()));
                continue;
            }

            ToolAction proposed = response.proposedAction();

            if (proposed == null) {
                history.add(AgentEvent.governorIntervention(
                        "HARD_BLOCK: LLM returned no action and no final answer."));
                continue;
            }

            if (regulationActive) {
                GovernorDecision decision = governor.reviewAction(controlSignal, history, proposed);
                if (decision.warning()) {
                    history.add(AgentEvent.governorIntervention("SOFT_WARNING: " + decision.reason()));
                }
                if (decision.blocked()) {
                    history.add(AgentEvent.governorIntervention("HARD_BLOCK: " + decision.reason()));
                    continue;
                }
            }

            boolean wasInduced = false;
            ToolResult result;
            if (induceNextToolCall) {
                result = ToolResult.failure(
                        "TRANSIENT_READ_ERROR",
                        "Control-generated disturbance: regulatory tightening corrupted this action.");
                inducedFailureCount++;
                induceNextToolCall = false;
                wasInduced = true;
            } else {
                result = tools.execute(proposed);
            }

            if (costAware) {
                inducedRate = (1.0 - backoffLambda) * inducedRate + backoffLambda * (wasInduced ? 1.0 : 0.0);
            }

            history.add(AgentEvent.toolCall(proposed, result));
            governor.observe(proposed, result);
        }

        return "Stopped: maximum number of steps reached.";
    }

    public List<AgentEvent> history() {
        return history;
    }

    public List<RegulatorySignal> controlSignalHistory() {
        return controlSignalHistory;
    }

    public List<Double> regulationIntensityHistory() {
        return regulationIntensityHistory;
    }

    public int inducedFailureCount() {
        return inducedFailureCount;
    }

    public int backoffStepCount() {
        return backoffStepCount;
    }
}
