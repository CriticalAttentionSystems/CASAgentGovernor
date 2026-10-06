/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package org.cas.agentgovernor.agents;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Deque;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Set;

import org.cas.agentgovernor.core.AgentEvent;
import org.cas.agentgovernor.core.CASGovernor;
import org.cas.agentgovernor.core.GovernorDecision;
import org.cas.agentgovernor.core.RegulatorySignal;
import org.cas.agentgovernor.core.ToolAction;
import org.cas.agentgovernor.core.ToolResult;
import org.cas.agentgovernor.policy.ActionPolicy;
import org.cas.agentgovernor.policy.ActionPolicy.PolicyDecision;
import org.cas.agentgovernor.tools.ToolRegistry;

/**
 * Discrete stochastic tool-use agent supporting a PERIODIC RF/DF dwell
 * schedule within a single episode -- direct discrete-substrate test of
 * "Intermittent Control Is Not Diluted Control" (paper 3): does a fixed
 * total budget of anticipatory (RF) regulation perform differently when
 * scheduled as many short interruptions versus one long contiguous block,
 * genuinely differently from what an occupancy-weighted mixture of fixed
 * RF and fixed DF outcomes would predict?
 *
 * <p>Additive, not a modification of {@link StochasticToolAgent}: that
 * class's regulationLagSteps is fixed for the whole episode by design,
 * matching every existing pre-registered result built on it, and stays
 * completely untouched. This class reuses its exact RF/DF mechanics (see
 * that class's own extensive documentation on the pending-queue deadlock
 * fix) but makes the RF-vs-DF choice a function of the CURRENT step
 * against a periodic schedule, matching paper 3's exact protocol:
 * {@code RF_L -> DF_L -> RF_L -> DF_L -> ...}.
 *
 * <h2>How the lagged view is handled across dwell transitions</h2>
 * On RF steps, the pending-queue/lag mechanism is bypassed entirely --
 * identical to StochasticToolAgent's own regulationLagSteps==0 behavior:
 * the real-time observed signal and full real history drive everything.
 * On transitioning INTO a DF dwell block, {@code visibleToGovernor} is
 * rebuilt from the real history MINUS the tool calls from the most recent
 * {@code dfLagSteps} steps, which are moved into the pending queue with
 * their original step indices. The lagged governor therefore does not
 * forget what happened before DF started -- it keeps everything old
 * enough to have cleared the lag window -- but the lag is in force from
 * the block's FIRST step.
 *
 * <p>An earlier version instead caught {@code visibleToGovernor} up to a
 * full copy of real-time history on entering DF. Because
 * {@link CASGovernor#evaluate} is a pure function of the list it is given,
 * that made the first step of every DF block behave EXACTLY like an RF
 * step: a block of length L contained only L-1 genuinely lagged steps, and
 * at L=1 the agent was byte-for-byte identical to a pure-RF run while
 * still reporting ~50% DF occupancy in modeHistory. Any occupancy-weighted
 * comparison built on that reported occupancy was therefore measuring
 * nothing. Seeding the queue at the transition is what makes reported and
 * effective occupancy agree.
 *
 * <p>On transitioning back to RF, any items still queued are simply
 * abandoned (RF bypasses the queue and uses real-time history directly, so
 * they would never have been consulted anyway); the next DF block reseeds
 * the queue from the real history again.
 *
 * <p>Reference/null model for interpreting results: paper 3's own
 * occupancy-weighted mixture, {@code predicted = p_RF * metric_RF_fixed +
 * (1-p_RF) * metric_DF_fixed}, built from ALREADY-COLLECTED pure-governed
 * (RF, lag=0) and pure-disturbance_first (DF, fixed lag) data from the
 * original synthetic sweep -- see the analysis script for how this
 * comparison is computed. This class only produces the switching
 * trajectories; it does not compute the reference itself.
 *
 * @author veronique
 */
public class SwitchingStochasticToolAgent {

    private final ActionPolicy policy;
    private final ToolRegistry tools;
    private final CASGovernor governor;
    private final boolean reweight;
    private final boolean enforceBlocking;
    private final int dwellLength;
    private final int dfLagSteps;

    private final List<AgentEvent> history = new ArrayList<>();
    private final List<RegulatorySignal> controlSignalHistory = new ArrayList<>();
    private final List<Double> entropyHistory = new ArrayList<>();

    /** Per-step record of which mode was actually active -- exposed for diagnostics/plots. */
    private final List<String> modeHistory = new ArrayList<>();

    public SwitchingStochasticToolAgent(
            ActionPolicy policy,
            ToolRegistry tools,
            CASGovernor governor,
            boolean reweight,
            boolean enforceBlocking,
            int dwellLength,
            int dfLagSteps
    ) {
        this.policy = policy;
        this.tools = tools;
        this.governor = governor;
        this.reweight = reweight;
        this.enforceBlocking = enforceBlocking;
        this.dwellLength = Math.max(1, dwellLength);
        this.dfLagSteps = Math.max(1, dfLagSteps);
    }

    public String run(String userTask, int maxSteps) {
        List<AgentEvent> visibleToGovernor = new ArrayList<>();
        Deque<PendingEvent> pendingToolCallEvents = new ArrayDeque<>();

        // Every tool call of the episode, tagged with the step it occurred
        // at, recorded in BOTH modes: an RF->DF transition needs to know
        // which of the preceding steps' tool calls are still inside the lag
        // window, and those may well have happened during the RF block.
        List<PendingEvent> toolCallLog = new ArrayList<>();

        boolean wasRFLastStep = true; // episode starts in RF, matching schedule block 0

        for (int step = 0; step < maxSteps; step++) {
            boolean isRF = (step / dwellLength) % 2 == 0;
            modeHistory.add(isRF ? "RF" : "DF");

            // Observational signal: ALWAYS computed from the real, unlagged,
            // real-time history -- same rationale as every other agent in
            // this codebase: this is what feeds the metrics columns and
            // must stay comparable regardless of which mode is active.
            RegulatorySignal observedSignal = governor.evaluate(history);
            history.add(AgentEvent.regulatorySignal(observedSignal));

            if (isRF && !wasRFLastStep) {
                // Just transitioned DF -> RF: abandon whatever was still
                // queued (RF bypasses it entirely from here).
                pendingToolCallEvents.clear();
            }

            if (!isRF && wasRFLastStep) {
                // Just transitioned RF -> DF: rebuild the lagged view as
                // everything except the tool calls still inside the lag
                // window, which go back on the queue. DF does not erase
                // pre-DF knowledge, but it is lagged from step one.
                visibleToGovernor = seedLaggedView(step, toolCallLog, pendingToolCallEvents);
            }

            if (!isRF) {
                // Release any pending tool-call events whose lag window has
                // elapsed, purely as a function of step count -- see
                // StochasticToolAgent's own extensive comment on why this
                // must not be gated behind a new tool call arriving.
                while (!pendingToolCallEvents.isEmpty()
                        && step - pendingToolCallEvents.peekFirst().stepIndex() > dfLagSteps) {
                    visibleToGovernor.add(pendingToolCallEvents.removeFirst().event());
                }
            }

            RegulatorySignal controlSignal;
            List<AgentEvent> governorView;
            if (isRF) {
                controlSignal = observedSignal;
                governorView = history;
            } else {
                controlSignal = governor.evaluate(visibleToGovernor);
                governorView = visibleToGovernor;
            }
            controlSignalHistory.add(controlSignal);

            PolicyDecision policyDecision = policy.choose(
                    userTask,
                    history,
                    controlSignal,
                    reweight
            );

            entropyHistory.add(policyDecision.entropy());

            if (policyDecision.wantsFinalAnswer()) {
                if (!enforceBlocking) {
                    history.add(AgentEvent.finalAnswer(policyDecision.finalAnswer()));
                    return policyDecision.finalAnswer();
                }

                GovernorDecision finalDecision =
                        governor.reviewFinalAnswer(controlSignal, governorView, policyDecision.finalAnswer());

                if (finalDecision.allowed()) {
                    history.add(AgentEvent.finalAnswer(policyDecision.finalAnswer()));
                    return policyDecision.finalAnswer();
                }

                history.add(AgentEvent.governorIntervention(
                        "HARD_BLOCK: " + finalDecision.reason()
                ));
                wasRFLastStep = isRF;
                continue;
            }

            ToolAction proposed = policyDecision.action();

            if (proposed == null) {
                history.add(AgentEvent.governorIntervention(
                        "HARD_BLOCK: stochastic policy produced no action."
                ));
                wasRFLastStep = isRF;
                continue;
            }

            if (enforceBlocking) {
                GovernorDecision decision = governor.reviewAction(controlSignal, governorView, proposed);

                if (decision.warning()) {
                    history.add(AgentEvent.governorIntervention(
                            "SOFT_WARNING: " + decision.reason()
                    ));
                }

                if (decision.blocked()) {
                    history.add(AgentEvent.governorIntervention(
                            "HARD_BLOCK: " + decision.reason()
                    ));
                    wasRFLastStep = isRF;
                    continue;
                }
            }

            ToolResult result = tools.execute(proposed);
            AgentEvent toolCallEvent = AgentEvent.toolCall(proposed, result);
            history.add(toolCallEvent);
            governor.observe(proposed, result);

            PendingEvent logged = new PendingEvent(step, toolCallEvent);
            toolCallLog.add(logged);

            if (!isRF) {
                pendingToolCallEvents.addLast(logged);
            }

            wasRFLastStep = isRF;
        }

        return "Stopped: maximum number of steps reached.";
    }

    /**
     * Builds the view a governor entering a DF block at {@code step} should
     * have, and refills {@code pending} with the tool calls it is not yet
     * allowed to see.
     *
     * <p>The release rule below is {@code step - stepIndex > dfLagSteps},
     * so a tool call from step {@code t} first becomes visible at step
     * {@code t + dfLagSteps + 1}. Inverting that at the transition: calls
     * from steps {@code >= step - dfLagSteps} are still in flight, and
     * everything older is visible. Iterating {@code toolCallLog} in step
     * order keeps the refilled queue ordered, as the release loop requires.
     */
    private List<AgentEvent> seedLaggedView(
            int step, List<PendingEvent> toolCallLog, Deque<PendingEvent> pending
    ) {
        int firstInFlightStep = step - dfLagSteps;

        pending.clear();
        Set<AgentEvent> inFlight = Collections.newSetFromMap(new IdentityHashMap<>());
        for (PendingEvent logged : toolCallLog) {
            if (logged.stepIndex() >= firstInFlightStep) {
                inFlight.add(logged.event());
                pending.addLast(logged);
            }
        }

        List<AgentEvent> view = new ArrayList<>();
        for (AgentEvent event : history) {
            if (!inFlight.contains(event)) {
                view.add(event);
            }
        }
        return view;
    }

    /** A tool-call event awaiting release into visibleToGovernor, tagged with the step it occurred at. */
    private record PendingEvent(int stepIndex, AgentEvent event) {
    }

    public List<AgentEvent> history() {
        return history;
    }

    public List<RegulatorySignal> controlSignalHistory() {
        return controlSignalHistory;
    }

    public List<Double> entropyHistory() {
        return entropyHistory;
    }

    /** Which mode ("RF" or "DF") was active at each step, in order. */
    public List<String> modeHistory() {
        return modeHistory;
    }
}
