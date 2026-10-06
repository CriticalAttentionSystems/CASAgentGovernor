/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package org.cas.agentgovernor.agents;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.List;

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
 * Discrete stochastic tool-use agent. Supports three axes of regulation:
 * how STRONG it is (reweight / enforceBlocking), how PROMPTLY it acts
 * relative to disturbances (regulationLagSteps, RF vs DF -- see prior
 * class doc), and, as of this addition, whether the signal driving it is
 * even CONTINGENT on this episode's real history at all
 * (scriptedControlSignals -- the sham/placebo condition).
 *
 * <p><b>Why the sham mode exists.</b> Every condition prior to this one
 * answers "how strong should regulation be" and "how promptly should it
 * act." Neither answers a more basic question: is the apparent benefit of
 * "governed" attributable to the signal actually TRACKING real disturbance
 * history, or would any similarly-strong, similarly-frequent perturbation
 * to the sampling weights produce the same effect regardless of whether it
 * corresponds to anything real? A governor that reweights on a schedule
 * decorrelated from actual failures, but with matched intervention
 * frequency and strength, is the standard placebo/sham control for this
 * question (analogous to sham stimulation controls in neuroscience, or
 * randomized-ablation controls in ML). If "governed" doesn't beat this
 * sham condition, the apparent regulation effect cannot be attributed to
 * contingency on real history -- it would be indistinguishable from a
 * randomization effect.
 *
 * <p>When {@code scriptedControlSignals} is supplied, the agent's control
 * signal at step {@code i} is simply {@code scriptedControlSignals.get(i)}
 * -- NOT computed from this episode's own history at all, lagged or
 * otherwise. The caller (see {@code RunStochasticPolicyExperiment}'s
 * {@code sham_regulation} condition) is responsible for constructing this
 * list by shuffling a real {@code governed} run's realized per-step signal
 * sequence, so that the sham run receives the exact same MULTISET of
 * regulatory states (matched total intervention strength/frequency) with
 * their step-to-step contingency destroyed. The agent's own discovery
 * logic and the governor's literal-repeat blocking still operate on this
 * run's REAL history throughout -- only the state-dependent portion of
 * regulation (which multipliers apply, whether RELEASED/recovery-mode
 * blocking fires) is driven by the scripted, decorrelated sequence.
 *
 * @author veronique
 */
public class StochasticToolAgent {

    private final ActionPolicy policy;
    private final ToolRegistry tools;
    private final CASGovernor governor;
    private final boolean reweight;
    private final boolean enforceBlocking;
    private final int regulationLagSteps;

    /**
     * If non-null, activates sham mode: the control signal at step i is
     * {@code scriptedControlSignals.get(i)} rather than anything computed
     * from this episode's own history. Takes priority over
     * {@code regulationLagSteps} when present. The run ends (with the
     * step-budget timeout message) if more steps are attempted than this
     * list has entries -- callers should size it to the intended episode
     * length exactly (see class doc).
     */
    private final List<RegulatorySignal> scriptedControlSignals;

    private final List<AgentEvent> history = new ArrayList<>();

    /** The control signal actually applied at each step, in order. Exposed so a
     *  {@code governed} (lag=0, unscripted) run's realized sequence can be
     *  captured and later fed into a paired sham run as
     *  {@code scriptedControlSignals}. */
    private final List<RegulatorySignal> controlSignalHistory = new ArrayList<>();

    /**
     * The Shannon entropy of the policy's actual sampling distribution at
     * each step, in order -- see {@code ActionPolicy.PolicyDecision#entropy()}
     * for what this literally is (a real S_vN-analogue, not a metaphor).
     * Recorded regardless of condition (reweight/enforceBlocking/sham/DF),
     * since the weighted-choice distribution exists and is sampled from
     * every step under every condition -- this makes entropy directly
     * comparable across all seven conditions in
     * RunStochasticPolicyExperiment.
     */
    private final List<Double> entropyHistory = new ArrayList<>();

    public StochasticToolAgent(
            ActionPolicy policy,
            ToolRegistry tools,
            CASGovernor governor,
            boolean reweight,
            boolean enforceBlocking
    ) {
        this(policy, tools, governor, reweight, enforceBlocking, 0, null);
    }

    public StochasticToolAgent(
            ActionPolicy policy,
            ToolRegistry tools,
            CASGovernor governor,
            boolean reweight,
            boolean enforceBlocking,
            int regulationLagSteps
    ) {
        this(policy, tools, governor, reweight, enforceBlocking, regulationLagSteps, null);
    }

    /**
     * @param scriptedControlSignals see the class doc and field doc above. Pass
     *                               {@code null} for normal (non-sham) operation;
     *                               {@code regulationLagSteps} is ignored whenever
     *                               this is non-null.
     */
    public StochasticToolAgent(
            ActionPolicy policy,
            ToolRegistry tools,
            CASGovernor governor,
            boolean reweight,
            boolean enforceBlocking,
            int regulationLagSteps,
            List<RegulatorySignal> scriptedControlSignals
    ) {
        this.policy = policy;
        this.tools = tools;
        this.governor = governor;
        this.reweight = reweight;
        this.enforceBlocking = enforceBlocking;
        this.regulationLagSteps = Math.max(0, regulationLagSteps);
        this.scriptedControlSignals = scriptedControlSignals;
    }

    public String run(String userTask, int maxSteps) {
        List<AgentEvent> visibleToGovernor = new ArrayList<>();
        Deque<PendingEvent> pendingToolCallEvents = new ArrayDeque<>();

        boolean sham = scriptedControlSignals != null;

        for (int step = 0; step < maxSteps; step++) {
            // Observational signal: ALWAYS computed from the real, unlagged,
            // unscripted history and recorded -- identical rationale to the
            // RF/DF addition: this is what feeds the metrics columns and must
            // stay comparable across every condition, sham included.
            RegulatorySignal observedSignal = governor.evaluate(history);
            history.add(AgentEvent.regulatorySignal(observedSignal));

            if (!sham && regulationLagSteps > 0) {
                // Release any pending tool-call events whose lag window has
                // elapsed, purely as a function of STEP COUNT. This must NOT
                // be gated behind a new tool call arriving: if it is, the
                // lagged view can stall forever the moment the agent stops
                // calling tools -- which happens exactly when it (correctly,
                // per its own real history) believes it has the right final
                // answer. The lagged governor then never "catches up" to see
                // that success, blocks the answer as premature every step,
                // and nothing is left to ever advance the queue: a permanent
                // deadlock triggered precisely in the success case. Advancing
                // by elapsed steps instead bounds the stall to exactly
                // regulationLagSteps extra steps, as intended.
                while (!pendingToolCallEvents.isEmpty()
                        && step - pendingToolCallEvents.peekFirst().stepIndex() > regulationLagSteps) {
                    visibleToGovernor.add(pendingToolCallEvents.removeFirst().event());
                }
            }

            RegulatorySignal controlSignal;
            List<AgentEvent> governorView;
            if (sham) {
                if (step >= scriptedControlSignals.size()) {
                    // Scripted sequence exhausted before the agent finished --
                    // end the run the same way running out of maxSteps would.
                    return "Stopped: maximum number of steps reached.";
                }
                controlSignal = scriptedControlSignals.get(step);
                // Repeat-detection intentionally stays on REAL history even in
                // sham mode: a sham test asks whether STATE-CONTINGENT signal
                // timing matters, not whether literal-repeat blocking (already
                // isolated separately by block_only) should also be decorrelated.
                governorView = history;
            } else if (regulationLagSteps == 0) {
                controlSignal = observedSignal;
                governorView = history;
            } else {
                controlSignal = governor.evaluate(visibleToGovernor);
                // Under DF, repeat-detection lags along with the signal itself --
                // the controller's whole view of the episode is delayed, not just
                // its state label.
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
                continue;
            }

            ToolAction proposed = policyDecision.action();

            if (proposed == null) {
                history.add(AgentEvent.governorIntervention(
                        "HARD_BLOCK: stochastic policy produced no action."
                ));
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
                    continue;
                }
            }

            ToolResult result = tools.execute(proposed);
            AgentEvent toolCallEvent = AgentEvent.toolCall(proposed, result);
            history.add(toolCallEvent);
            governor.observe(proposed, result);

            if (!sham && regulationLagSteps > 0) {
                pendingToolCallEvents.addLast(new PendingEvent(step, toolCallEvent));
            }
        }

        return "Stopped: maximum number of steps reached.";
    }

    /** A tool-call event awaiting release into visibleToGovernor, tagged with the step it occurred at. */
    private record PendingEvent(int stepIndex, AgentEvent event) {
    }

    public List<AgentEvent> history() {
        return history;
    }

    /** The control signal actually applied at each step, in order (see field doc). */
    public List<RegulatorySignal> controlSignalHistory() {
        return controlSignalHistory;
    }

    /** The policy's sampling-distribution Shannon entropy at each step, in order (see field doc). */
    public List<Double> entropyHistory() {
        return entropyHistory;
    }
}
