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

import org.iram.omega.iram_omega_q.cognition.Hamiltonian;
import org.iram.omega.iram_omega_q.cognition.QuantumCognitiveState;
import org.iram.omega.iram_omega_q.cognition.QuantumRegulationAgent;
import org.iram.omega.iram_omega_q.control.MuController;

/**
 * Real-LLM-backed agent, structurally identical to {@link GovernedLLMAgent}
 * (same LLMClient, same prompt-injection-plus-blocking role for the
 * regulatory signal), but with the SIGNAL COMPUTATION replaced: instead of
 * {@code CASGovernor.evaluate()}'s discrete 5-state lookup table, the
 * signal each step is derived from a real {@code QuantumRegulationAgent} --
 * genuine {@code Hamiltonian} evolution, a genuine {@code MuController}, and
 * real von Neumann entropy.
 *
 * <h2>What stays discrete, and why</h2>
 * Unlike the T005 proof of concept, this does NOT use Born-rule
 * {@code measure()} to pick the LLM's actions -- an LLM doesn't have a
 * fixed "basis" of candidate actions to sample from the way a 3-file pick
 * did, and replacing free-form LLM reasoning with quantum measurement would
 * be a much bigger, less defensible leap than what's actually needed here.
 * The LLM still proposes its own actions in free text, exactly as in
 * {@code GovernedLLMAgent}. What the quantum layer replaces is narrower and
 * more defensible: computing WHICH REGULATORY STATE to report and act on.
 *
 * <p>Two things are deliberately still borrowed from {@code CASGovernor}
 * rather than reinvented:
 * <ul>
 *   <li><b>RELEASED stays grounded in real history, not quantum readout.</b>
 *       Whether "useful content has been recovered" is a FACT about whether
 *       a real, relevant, successful read happened -- the continuous layer
 *       has no business inventing that. Every step, the real observational
 *       signal ({@code governor.evaluate(history)}) is computed first; if
 *       it says RELEASED, that's used as-is. Only when it does NOT say
 *       RELEASED does the quantum-derived signal take over, choosing among
 *       NORMAL/DISTURBED/RECOVERING/STUCK. NOTE: that observational signal
 *       is ALSO what gets logged to {@code history()}'s REGULATORY_SIGNAL
 *       entries for transcript/bookkeeping purposes (matching every other
 *       agent in this codebase) -- it is NOT the same thing as what
 *       actually drove the LLM's prompt or blocking decisions. For that,
 *       read {@link #controlSignalHistory()}, not {@code history()}. An
 *       early version of this class's own demo runner's printed transcript
 *       was misread this way once already -- see the class's revision
 *       history in chat for the full diagnosis.</li>
 *   <li><b>Repeat-action blocking and premature-answer blocking are still
 *       CASGovernor's real, tested logic</b> (via the explicit-signal
 *       {@code reviewAction}/{@code reviewFinalAnswer} overloads already
 *       used by the sham/DF machinery) -- only the SIGNAL passed into them
 *       is quantum-derived when not RELEASED, the blocking rules themselves
 *       are unchanged. In particular, repeat-detection reads real
 *       {@code history} directly and does not consult the signal's state
 *       at all, so it works correctly even while the quantum signal is
 *       misconfigured -- which is exactly what happened in the run that
 *       prompted this recalibration.</li>
 * </ul>
 *
 * <h2>The continuous-to-discrete mapping (recalibrated once already; still
 * hand-picked, still not fully validated)</h2>
 * The quantum state is a minimal 2-dimensional abstract "on-track vs.
 * off-track" basis (NOT one basis element per candidate file -- there's no
 * fixed candidate set for a general LLM task the way T005 had one), started
 * at the deterministic {@code |0>} state (index 0 = "on-track"), NOT a
 * salience-weighted coherent state like T005 used.
 *
 * <p><b>First calibration attempt was wrong, and here's why:</b> the first
 * version copied T005's {@code targetEntropy = 0.3} unchanged. T005 used a
 * 3-dimensional, salience-initialized state with real entropy headroom; this
 * agent's 2-dimensional, deterministic-start state never got anywhere near
 * 0.3 entropy even during a real, confirmed tool failure (observed max:
 * 0.055 in one real episode). Since {@code MuController.update()}'s error
 * term is {@code beta * (entropy - targetEntropy)}, an unreachable target
 * meant that term was persistently negative regardless of what was actually
 * happening, so mu just drifted toward its floor the entire episode no
 * matter what -- the quantum-derived signal stayed NORMAL throughout, even
 * immediately after a real failure. Retuned against that one real trace:
 * {@code targetEntropy = 0.01} sits between typical calm-period entropy
 * (0.0001-0.006 in that trace) and post-disturbance entropy (0.017-0.055),
 * so the error term now correctly flips positive (mu rises) during real
 * trouble and negative (mu relaxes) during calm stretches -- verified
 * against the actual recorded trace before being adopted, not guessed.
 * Thresholds below were re-set proportionally. This is still a first
 * REtuning off a single episode, not a validated calibration -- expect it
 * may need another pass once more real episodes exist.
 * <pre>
 *   STUCK       mu >= 0.28 AND entropy >= 0.015, sustained 3+ consecutive steps
 *   RECOVERING  mu >= 0.22
 *   DISTURBED   mu >= 0.205 OR entropy >= 0.008
 *   NORMAL      otherwise
 * </pre>
 *
 * <h2>Stabilization is now actually wired in</h2>
 * The first version never called {@code setMindfulnessFocus(...)} at all,
 * meaning the stabilization half of {@code QuantumRegulationAgent}'s
 * machinery was completely inert -- only the noise-suppression half (via
 * mu) was ever doing anything. This version calls
 * {@code setMindfulnessFocus(0)} once, unconditionally, at construction:
 * index 0 ("on-track") is the permanent attentional target the mindfulness
 * operator gently stabilizes toward whenever mu &gt; 0, matching the
 * framework's own description of mindfulness as a continuous, low-level
 * stabilizing force rather than something toggled on/off by discrete
 * events.
 *
 * <h2>How disturbance is injected</h2>
 * Same persistent-field caveat as the T005 pilot:
 * {@code setEmotionalNoise(double)} is read every subsequent {@code step()}
 * until changed again, so it is explicitly reset to 0 and only raised for
 * exactly one following step after ANY real signal of trouble this step:
 * a tool-call failure, a hard-blocked action, or a hard-blocked final
 * answer.
 *
 * @author veronique
 */
public class QuantumGovernedLLMAgent {

    private final LLMClient llm;
    private final CASGovernor governor;
    private final ToolRegistry tools;
    private final List<AgentEvent> history = new ArrayList<>();
    private final long delayMillis;

    private final QuantumRegulationAgent quantumAgent;

    private final List<RegulatorySignal> controlSignalHistory = new ArrayList<>();
    private final List<Double> entropyHistory = new ArrayList<>();
    private final List<Double> muHistory = new ArrayList<>();

    // ---- Hand-picked, retuned-once-against-real-data constants -- see class doc. ----
    private static final double HAMILTONIAN_COUPLING = 0.6;
    private static final double HAMILTONIAN_LENGTH_SCALE = 1.5;
    private static final double MU_INIT = 0.2;
    private static final double MU_MIN = 0.0;
    private static final double MU_MAX = 0.95;
    private static final double MU_ALPHA0 = 0.4;
    private static final double MU_BETA0 = 0.6;
    private static final double MU_TARGET_ENTROPY = 0.01; // was 0.3 -- see class doc
    private static final double DISTURBANCE_ON_TROUBLE = 0.35;

    private static final double STUCK_MU_THRESHOLD = 0.28; // was 0.30
    private static final double STUCK_ENTROPY_THRESHOLD = 0.015; // was 0.20
    private static final int STUCK_CONSECUTIVE_STEPS_REQUIRED = 3;
    private static final double RECOVERING_MU_THRESHOLD = 0.22; // was 0.30
    private static final double DISTURBED_MU_THRESHOLD = 0.205; // was 0.22
    private static final double DISTURBED_ENTROPY_THRESHOLD = 0.008; // was 0.10

    /** Index of the "on-track" basis state -- see class doc on stabilization. */
    private static final int ON_TRACK_FOCUS_INDEX = 0;

    private int consecutiveStuckReadings = 0;

    public QuantumGovernedLLMAgent(
            LLMClient llm,
            CASGovernor governor,
            ToolRegistry tools,
            long delayMillis,
            long quantumSeed
    ) {
        this.llm = llm;
        this.governor = governor;
        this.tools = tools;
        this.delayMillis = delayMillis;

        QuantumCognitiveState psi = new QuantumCognitiveState(2);
        double[] energies = new double[2];
        Hamiltonian hamiltonian = Hamiltonian.attentional(energies, HAMILTONIAN_COUPLING, HAMILTONIAN_LENGTH_SCALE);
        MuController muController = new MuController(
                MU_INIT, MU_MIN, MU_MAX, MU_ALPHA0, MU_BETA0, MU_TARGET_ENTROPY
        );

        this.quantumAgent = new QuantumRegulationAgent(psi, hamiltonian, muController, new Random(quantumSeed));
        this.quantumAgent.setControlOrdering(QuantumRegulationAgent.ControlOrdering.REGULATION_FIRST);
        this.quantumAgent.setMindfulnessFocus(ON_TRACK_FOCUS_INDEX);
    }

    public String run(String userTask, int maxSteps) throws Exception {
        double pendingNoise = 0.0;

        for (int step = 0; step < maxSteps; step++) {
            // Observational signal: always the REAL, discrete CASGovernor
            // reading, regardless of what the quantum layer says -- this is
            // what grounds RELEASED in an actual fact, and is what any
            // future metrics/analysis of regulatory-state counts should
            // read, consistent with every other agent in this codebase.
            RegulatorySignal observedSignal = governor.evaluate(history);
            history.add(AgentEvent.regulatorySignal(observedSignal));

            quantumAgent.setEmotionalNoise(pendingNoise);
            pendingNoise = 0.0;
            quantumAgent.step();

            double entropy = QuantumCognitiveState.vonNeumannEntropy(quantumAgent.state());
            double mu = quantumAgent.mindfulnessLevel();
            entropyHistory.add(entropy);
            muHistory.add(mu);

            RegulatorySignal controlSignal = observedSignal.isReleased()
                    ? observedSignal
                    : deriveQuantumSignal(entropy, mu, observedSignal.burden());

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

                history.add(AgentEvent.governorIntervention("HARD_BLOCK: " + finalDecision.reason()));
                pendingNoise = DISTURBANCE_ON_TROUBLE;
                continue;
            }

            ToolAction proposed = response.proposedAction();

            if (proposed == null) {
                history.add(AgentEvent.governorIntervention(
                        "HARD_BLOCK: LLM returned no action and no final answer."));
                pendingNoise = DISTURBANCE_ON_TROUBLE;
                continue;
            }

            GovernorDecision decision = governor.reviewAction(controlSignal, history, proposed);

            if (decision.warning()) {
                history.add(AgentEvent.governorIntervention("SOFT_WARNING: " + decision.reason()));
            }

            if (decision.blocked()) {
                history.add(AgentEvent.governorIntervention("HARD_BLOCK: " + decision.reason()));
                pendingNoise = DISTURBANCE_ON_TROUBLE;
                continue;
            }

            ToolResult result = tools.execute(proposed);
            history.add(AgentEvent.toolCall(proposed, result));
            governor.observe(proposed, result);

            if (!result.success()) {
                pendingNoise = DISTURBANCE_ON_TROUBLE;
            }
        }

        return "Stopped: maximum number of steps reached.";
    }

    /**
     * Maps continuous (entropy, mu) into a discrete RegulatorySignal, in
     * descending-severity priority order. See class doc for the exact
     * thresholds and why they are hand-picked, not validated.
     */
    private RegulatorySignal deriveQuantumSignal(double entropy, double mu, int observedBurden) {
        boolean muHigh = mu >= STUCK_MU_THRESHOLD;
        boolean entropyHigh = entropy >= STUCK_ENTROPY_THRESHOLD;

        if (muHigh && entropyHigh) {
            consecutiveStuckReadings++;
        } else {
            consecutiveStuckReadings = 0;
        }

        if (consecutiveStuckReadings >= STUCK_CONSECUTIVE_STEPS_REQUIRED) {
            return new RegulatorySignal(RegulatoryState.STUCK, observedBurden, String.format(
                    "Quantum regulatory readout: entropy=%.3f, mu=%.3f, sustained over %d steps. "
                            + "Repeated failure pattern detected. Stop repeating failed assumptions. "
                            + "Prefer inspection, schema/listing, or clarification.",
                    entropy, mu, consecutiveStuckReadings));
        }

        if (mu >= RECOVERING_MU_THRESHOLD) {
            return new RegulatorySignal(RegulatoryState.RECOVERING, observedBurden, String.format(
                    "Quantum regulatory readout: entropy=%.3f, mu=%.3f. Actively recovering from a recent "
                            + "disturbance. Avoid blind retry; take an uncertainty-reducing action.",
                    entropy, mu));
        }

        if (mu >= DISTURBED_MU_THRESHOLD || entropy >= DISTURBED_ENTROPY_THRESHOLD) {
            return new RegulatorySignal(RegulatoryState.DISTURBED, observedBurden, String.format(
                    "Quantum regulatory readout: entropy=%.3f, mu=%.3f. A recent action did not go as expected. "
                            + "Consider a different approach rather than repeating the same assumption.",
                    entropy, mu));
        }

        return new RegulatorySignal(RegulatoryState.NORMAL, observedBurden,
                "Normal operation. Proceed with the task.");
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

    public List<Double> muHistory() {
        return muHistory;
    }
}
