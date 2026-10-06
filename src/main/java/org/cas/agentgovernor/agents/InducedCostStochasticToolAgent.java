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
import org.cas.agentgovernor.policy.ActionPolicy;
import org.cas.agentgovernor.policy.ActionPolicy.PolicyDecision;
import org.cas.agentgovernor.tools.ToolRegistry;

/**
 * Discrete tool-use agent that gives the governor's own tightening an
 * induced cost -- a direct discrete-substrate test of paper 4's
 * "control-generated disturbance" claim ("Noise-Source Structure in
 * IRAM-Omega-Q"): regulation is not only a stabilizer, it can itself become
 * a disturbance source, and as the cost of tightening grows the system
 * passes through amplification, a stochastic transition band, and
 * eventually suppression.
 *
 * <h2>What "tightening" and "induced cost" mean discretely</h2>
 * Paper 4's induced disturbance is {@code eta_induced = alpha * max(0,
 * mu_after - mu_before)^2}: only POSITIVE controller tightening generates
 * cost, it is charged same-cycle, and relaxation is free. The discrete
 * governor tightens by escalating its regulatory state (NORMAL -> DISTURBED
 * -> RECOVERING -> STUCK), which {@link ActionPolicy#modulate} turns into a
 * progressively harder clamp on the sampling distribution. We therefore
 * define a scalar regulation intensity
 * <pre>
 *   g(NORMAL)=0.00  g(DISTURBED)=0.75  g(RECOVERING)=0.85  g(STUCK)=0.95  g(RELEASED)=0.99
 * </pre>
 * (equal to {@code 1 - retryMultiplier} in {@code modulate}, i.e. how hard
 * retry is suppressed), and positive tightening {@code dgPlus = max(0, g(t)
 * - g(t-1))}. After the control signal is computed each step, an induced
 * failure is drawn with probability {@code min(1, alpha * dgPlus^2)}. If it
 * fires, the NEXT tool execution is forced to return a TRANSIENT failure
 * regardless of whether the action would really have succeeded -- "the
 * governor's abrupt clamp corrupted the agent's next move."
 *
 * <h2>Why the feedback loop paper 4 needs already exists</h2>
 * An injected induced failure is a real failed tool call in {@code history}.
 * The governor sees more failures, escalates its state, tightens harder,
 * raises {@code dgPlus}, and induces still more failures. That positive
 * feedback is exactly the path that produces paper 4's bifurcation, so the
 * discrete substrate can genuinely exhibit (or fail to exhibit) it.
 *
 * <h2>Cost-blind vs cost-aware governor (phase 1 vs phase 2)</h2>
 * In the default <b>cost-blind</b> mode ({@code costAware == false}), the
 * governor always escalates on failures; it does not know tightening is now
 * expensive and cannot back off. That mode reproduces paper 4's
 * amplification and a runaway collapse, but NOT paper 4's graceful
 * suppression (where regulation deliberately falls because tightening costs
 * too much), and it produces no bistable transition band.
 *
 * <p>In <b>cost-aware</b> mode ({@code costAware == true}, phase 2) the agent
 * maintains an EWMA of the recent induced-failure rate and, each step,
 * applies the governor's regulation (both reweighting AND blocking) only with
 * probability {@code pApply = 1 - min(1, backoffK * inducedRate)}. When its
 * own tightening has been generating failures, the governor goes hands-off
 * more often: intensity drops to 0 those steps, no new induced cost is
 * generated, the induced rate decays, and pApply recovers. That negative
 * feedback is the discrete analog of paper 4's cost-aware controller backing
 * off, and its stochastic per-step nature is what can split seeds into a
 * regulated basin and a suppressed basin -- the transition-band signature the
 * cost-blind mode lacks. It also tests the design-relevant question: does
 * letting the governor back off RESCUE the success the cost-blind governor
 * destroyed at high alpha?
 *
 * <p>At {@code alpha == 0} in cost-blind mode this class reduces exactly to a
 * plain {@link StochasticToolAgent} run at the given regulationLagSteps (RF
 * when lag 0, DF when lag 1), the intended clean regression check.
 * StochasticToolAgent itself stays completely untouched. (Cost-aware mode is
 * a no-op at alpha 0 too, since with no induced failures pApply stays 1.)
 *
 * @author veronique
 */
public class InducedCostStochasticToolAgent {

    /**
     * How the governor's positive tightening is turned into an induced cost.
     *
     * <ul>
     *   <li>{@code RESULT_INJECTION} (default): the next tool call is forced to
     *   return a transient failure, bypassing the tool. Simple and direct, but
     *   the failure is engineered by construction.</li>
     *   <li>{@code ARGUMENT_CORRUPTION}: the next tool call's target is
     *   perturbed by a corruption suffix and the REAL tool executes the
     *   degraded action, returning its own genuine not-found error rather than
     *   an injected result. The failed wrong-path action is what enters
     *   history, so the governor's burden and repeat logic operate on a real
     *   degraded action, not a synthetic transient. Used to show the feedback
     *   loop reproduces under a channel that degrades action quality rather
     *   than hardcoding an outcome. A softer corruption that leaves the target
     *   valid (truncating to a still-listable parent) attenuates the loop,
     *   since the cost must yield governor-visible failures to drive
     *   escalation.</li>
     * </ul>
     */
    public enum CostMechanism {
        RESULT_INJECTION,
        ARGUMENT_CORRUPTION
    }

    /** Backoff regime for the governor's own intervention. */
    public enum BackoffMode {
        /** Cost-blind: regulation always applied. */
        NONE,
        /** Cost-aware: apply with probability 1 - min(1, k*inducedRate) (adaptive). */
        ADAPTIVE,
        /** Weak-governor control: apply with a FIXED probability, non-adaptive. */
        FIXED
    }

    private final ActionPolicy policy;
    private final ToolRegistry tools;
    private final CASGovernor governor;
    private final boolean reweight;
    private final boolean enforceBlocking;
    private final int regulationLagSteps;
    private final double alpha;
    private final Random inducedRandom;

    private final BackoffMode backoffMode;
    private final double backoffLambda;        // EWMA weight for the induced-failure rate (ADAPTIVE)
    private final double backoffK;             // sensitivity: how fast pApply falls with rate (ADAPTIVE)
    private final double fixedApplyProbability; // constant apply probability (FIXED)
    private final CostMechanism costMechanism;

    private final List<AgentEvent> history = new ArrayList<>();
    private final List<RegulatorySignal> controlSignalHistory = new ArrayList<>();
    private final List<Double> entropyHistory = new ArrayList<>();

    /** Regulation intensity g(t) actually applied at each step, in order. */
    private final List<Double> regulationIntensityHistory = new ArrayList<>();

    private int inducedFailureCount = 0;
    /** Steps on which backoff (adaptive or fixed) made the governor go hands-off. */
    private int backoffStepCount = 0;
    /** Steps on which the governor actually applied its regulation -- realized intervention rate. */
    private int regulationActiveCount = 0;

    /** Cost-blind constructor (phase 1). Preserves the original behavior exactly. */
    public InducedCostStochasticToolAgent(
            ActionPolicy policy,
            ToolRegistry tools,
            CASGovernor governor,
            boolean reweight,
            boolean enforceBlocking,
            int regulationLagSteps,
            double alpha,
            Random inducedRandom
    ) {
        this(policy, tools, governor, reweight, enforceBlocking, regulationLagSteps,
                alpha, inducedRandom, false, 0.0, 0.0);
    }

    /**
     * Phase-2 constructor: {@code costAware} selects NONE vs ADAPTIVE backoff,
     * with the default {@code RESULT_INJECTION} cost mechanism. Preserved so the
     * phase-1/2 harnesses are byte-identical.
     */
    public InducedCostStochasticToolAgent(
            ActionPolicy policy,
            ToolRegistry tools,
            CASGovernor governor,
            boolean reweight,
            boolean enforceBlocking,
            int regulationLagSteps,
            double alpha,
            Random inducedRandom,
            boolean costAware,
            double backoffLambda,
            double backoffK
    ) {
        this(policy, tools, governor, reweight, enforceBlocking, regulationLagSteps, alpha, inducedRandom,
                costAware ? BackoffMode.ADAPTIVE : BackoffMode.NONE,
                backoffLambda, backoffK, -1.0, CostMechanism.RESULT_INJECTION);
    }

    /**
     * Full constructor exposing the cost mechanism and the backoff regime,
     * including the {@code FIXED} weak-governor control (constant
     * {@code fixedApplyProbability}, non-adaptive) used to test whether
     * cost-aware backoff is genuinely adaptive rather than merely equivalent to
     * intervening less often.
     */
    public InducedCostStochasticToolAgent(
            ActionPolicy policy,
            ToolRegistry tools,
            CASGovernor governor,
            boolean reweight,
            boolean enforceBlocking,
            int regulationLagSteps,
            double alpha,
            Random inducedRandom,
            BackoffMode backoffMode,
            double backoffLambda,
            double backoffK,
            double fixedApplyProbability,
            CostMechanism costMechanism
    ) {
        this.policy = policy;
        this.tools = tools;
        this.governor = governor;
        this.reweight = reweight;
        this.enforceBlocking = enforceBlocking;
        this.regulationLagSteps = Math.max(0, regulationLagSteps);
        this.alpha = Math.max(0.0, alpha);
        this.inducedRandom = inducedRandom;
        this.backoffMode = backoffMode == null ? BackoffMode.NONE : backoffMode;
        this.backoffLambda = backoffLambda;
        this.backoffK = backoffK;
        this.fixedApplyProbability = fixedApplyProbability;
        this.costMechanism = costMechanism == null ? CostMechanism.RESULT_INJECTION : costMechanism;
    }

    /** Regulation intensity ladder; see class doc. Equals 1 - retryMultiplier(state). */
    private static double intensityOf(RegulatoryState state) {
        return switch (state) {
            case NORMAL -> 0.00;
            case DISTURBED -> 0.75;
            case RECOVERING -> 0.85;
            case STUCK -> 0.95;
            case RELEASED -> 0.99;
        };
    }

    public String run(String userTask, int maxSteps) {
        // Lagged view for DF ordering, identical mechanics to
        // StochasticToolAgent (kept simple here: lag is 0 or 1 in the
        // induced-cost sweep, matching governed / disturbance_first).
        List<AgentEvent> visibleToGovernor = new ArrayList<>();
        java.util.Deque<PendingEvent> pendingToolCallEvents = new java.util.ArrayDeque<>();

        double previousIntensity = 0.0; // episode starts un-tightened (NORMAL)
        boolean induceNextToolCall = false;
        double inducedRate = 0.0; // EWMA of the per-tool-call induced-failure indicator

        for (int step = 0; step < maxSteps; step++) {
            RegulatorySignal observedSignal = governor.evaluate(history);
            history.add(AgentEvent.regulatorySignal(observedSignal));

            if (regulationLagSteps > 0) {
                while (!pendingToolCallEvents.isEmpty()
                        && step - pendingToolCallEvents.peekFirst().stepIndex() > regulationLagSteps) {
                    visibleToGovernor.add(pendingToolCallEvents.removeFirst().event());
                }
            }

            RegulatorySignal controlSignal;
            List<AgentEvent> governorView;
            if (regulationLagSteps == 0) {
                controlSignal = observedSignal;
                governorView = history;
            } else {
                controlSignal = governor.evaluate(visibleToGovernor);
                governorView = visibleToGovernor;
            }
            controlSignalHistory.add(controlSignal);

            // --- Backoff decision. NONE (cost-blind): always apply. ADAPTIVE
            // (cost-aware): apply with pApply = 1 - min(1, k*inducedRate), so the
            // governor stands down more when its own tightening has been costly.
            // FIXED (weak-governor control): apply with a constant probability,
            // NON-adaptive, to test whether adaptivity itself matters. ---
            boolean regulationActive = true;
            if (backoffMode == BackoffMode.ADAPTIVE) {
                double pApply = 1.0 - Math.min(1.0, backoffK * inducedRate);
                regulationActive = inducedRandom.nextDouble() < pApply;
            } else if (backoffMode == BackoffMode.FIXED) {
                regulationActive = inducedRandom.nextDouble() < fixedApplyProbability;
            }
            if (!regulationActive) {
                backoffStepCount++;
            } else {
                regulationActiveCount++;
            }
            boolean effectiveReweight = reweight && regulationActive;
            boolean effectiveBlocking = enforceBlocking && regulationActive;

            // --- Induced-cost accounting (paper 4). The clamp intensity the
            // governor is APPLYING this step drives the tightening cost. When
            // backed off, no clamp is applied, so intensity is 0 and no new
            // induced cost is generated. ---
            double intensity = effectiveReweight ? intensityOf(controlSignal.state()) : 0.0;
            regulationIntensityHistory.add(intensity);
            double dgPlus = Math.max(0.0, intensity - previousIntensity);
            previousIntensity = intensity;

            if (alpha > 0.0 && dgPlus > 0.0) {
                double inducedProbability = Math.min(1.0, alpha * dgPlus * dgPlus);
                if (inducedRandom.nextDouble() < inducedProbability) {
                    induceNextToolCall = true;
                }
            }

            PolicyDecision policyDecision = policy.choose(
                    userTask,
                    history,
                    controlSignal,
                    effectiveReweight
            );

            entropyHistory.add(policyDecision.entropy());

            if (policyDecision.wantsFinalAnswer()) {
                if (!effectiveBlocking) {
                    history.add(AgentEvent.finalAnswer(policyDecision.finalAnswer()));
                    return policyDecision.finalAnswer();
                }

                GovernorDecision finalDecision =
                        governor.reviewFinalAnswer(controlSignal, governorView, policyDecision.finalAnswer());

                if (finalDecision.allowed()) {
                    history.add(AgentEvent.finalAnswer(policyDecision.finalAnswer()));
                    return policyDecision.finalAnswer();
                }

                history.add(AgentEvent.governorIntervention("HARD_BLOCK: " + finalDecision.reason()));
                continue;
            }

            ToolAction proposed = policyDecision.action();

            if (proposed == null) {
                history.add(AgentEvent.governorIntervention(
                        "HARD_BLOCK: stochastic policy produced no action."));
                continue;
            }

            if (effectiveBlocking) {
                GovernorDecision decision = governor.reviewAction(controlSignal, governorView, proposed);

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
                wasInduced = true;
                inducedFailureCount++;
                induceNextToolCall = false;
                if (costMechanism == CostMechanism.RESULT_INJECTION) {
                    // Fabricate a transient failure, bypassing the tool.
                    result = ToolResult.failure(
                            "TRANSIENT_READ_ERROR",
                            "Control-generated disturbance: regulatory tightening corrupted this action.");
                } else {
                    // ARGUMENT_CORRUPTION: actually EXECUTE a corrupted (bad-path)
                    // version of the intended action, so the failure is the real
                    // tool's genuine not-found error, not a fabricated result. The
                    // failure is attributed to the agent's INTENDED action (below),
                    // modelling "the tightening corrupted the execution of the
                    // action you meant" -- which is what lets the governor's own
                    // repeat-blocking of that intended action drive the loop.
                    result = tools.execute(corruptTarget(proposed));
                }
            } else {
                result = tools.execute(proposed);
            }

            // The backoff EWMA tracks induced EVENTS (the tightening fired),
            // regardless of the mechanism.
            if (backoffMode == BackoffMode.ADAPTIVE) {
                inducedRate = (1.0 - backoffLambda) * inducedRate + backoffLambda * (wasInduced ? 1.0 : 0.0);
            }

            // Attribute the outcome to the agent's INTENDED action under both
            // mechanisms, so both engage the governor's repeat logic identically.
            AgentEvent toolCallEvent = AgentEvent.toolCall(proposed, result);
            history.add(toolCallEvent);
            governor.observe(proposed, result);

            if (regulationLagSteps > 0) {
                pendingToolCallEvents.addLast(new PendingEvent(step, toolCallEvent));
            }
        }

        return "Stopped: maximum number of steps reached.";
    }

    /**
     * ARGUMENT_CORRUPTION channel: perturb a proposed action's target with a
     * corruption suffix so it names a path that does not exist, and let the REAL
     * tool execute it. The tool returns its own genuine not-found error (not an
     * injected result), and the failed wrong-path action is what enters history.
     * This degrades action quality rather than hardcoding an outcome, so the
     * collapse is not engineered by the injection -- yet it still yields the
     * governor-visible failure that drives the feedback loop.
     */
    private static ToolAction corruptTarget(ToolAction action) {
        String target = action.target();
        String base = (target == null || target.isBlank()) ? "." : target.trim();
        while (base.endsWith("/") || base.endsWith("\\")) {
            base = base.substring(0, base.length() - 1);
        }
        return new ToolAction(action.tool(), action.actionType(), base + "__induced_corrupt", action.payload());
    }

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

    /** Regulation intensity g(t) applied at each step (the mean_mu analog). */
    public List<Double> regulationIntensityHistory() {
        return regulationIntensityHistory;
    }

    /** Number of induced (control-generated) failures injected this episode. */
    public int inducedFailureCount() {
        return inducedFailureCount;
    }

    /** Steps on which backoff made the governor go hands-off (0 in cost-blind mode). */
    public int backoffStepCount() {
        return backoffStepCount;
    }

    /** Steps on which the governor actually applied its regulation (realized intervention count). */
    public int regulationActiveCount() {
        return regulationActiveCount;
    }
}
