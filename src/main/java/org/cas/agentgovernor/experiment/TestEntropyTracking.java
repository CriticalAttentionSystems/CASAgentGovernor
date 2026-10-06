/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package org.cas.agentgovernor.experiment;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.cas.agentgovernor.core.AgentEvent;
import org.cas.agentgovernor.core.RegulatorySignal;
import org.cas.agentgovernor.core.RegulatoryState;
import org.cas.agentgovernor.policy.ActionPolicy;
import org.cas.agentgovernor.policy.ActionPolicy.PolicyDecision;
import org.cas.agentgovernor.policy.PolicyParameters;

/**
 * Fast, direct smoke test for entropy tracking (see ActionPolicy.PolicyDecision#entropy()).
 * Exercises ActionPolicy.choose() against every RegulatoryState with an
 * EMPTY history (no prior tool calls), so the resulting entropy reflects
 * pure base-vs-modulated weight structure, nothing episode-specific --
 * runs in well under a second, no full sweep needed.
 *
 * What to check by hand against the printed output:
 *
 *   1. Every printed entropy should be &gt;= 0 (Shannon entropy is never
 *      negative) and &lt;= ln(number of active choices) for that step --
 *      this script prints both bounds so you can verify directly.
 *   2. RELEASED should show the LOWEST entropy of the five states: its
 *      multiplier table (see ActionPolicy.modulate()) pushes
 *      prematureAnswerWeight x10 while crushing every other weight to
 *      0.01-0.1x, so the resulting distribution should be heavily
 *      concentrated on one choice -- close to 0 entropy, not exactly 0
 *      (the other choices still have nonzero weight, just tiny).
 *   3. NORMAL should reproduce the archetype's raw, unmodulated base
 *      weights exactly (modulate() returns p unchanged for NORMAL) --
 *      this is the one state where entropy is a direct, uninterpreted
 *      read of PolicyParameters.persistent()'s own weight spread.
 *   4. RECOVERING and STUCK should sit somewhere between NORMAL and
 *      RELEASED -- both push weight toward inspect/read-candidate and
 *      away from retry/premature-answer, but less extremely than RELEASED.
 *
 * Run:
 *   mvn compile exec:java -Dexec.mainClass=org.cas.agentgovernor.experiment.TestEntropyTracking
 *
 * @author veronique
 */
public class TestEntropyTracking {

    public static void main(String[] args) {
        PolicyParameters base = PolicyParameters.persistent();
        List<AgentEvent> emptyHistory = new ArrayList<>();
        String userTask = "Read output/summary.csv and report the mean post-error burden.";

        System.out.println("Smoke test: ActionPolicy entropy computation");
        System.out.println("Archetype: persistent, empty history (no prior tool calls)");
        System.out.println("=".repeat(70));

        List<RegulatoryState> states = List.of(
                RegulatoryState.NORMAL,
                RegulatoryState.DISTURBED,
                RegulatoryState.RECOVERING,
                RegulatoryState.STUCK,
                RegulatoryState.RELEASED
        );

        double[] entropies = new double[states.size()];

        for (int i = 0; i < states.size(); i++) {
            RegulatoryState state = states.get(i);
            RegulatorySignal signal = new RegulatorySignal(state, 0, "smoke test signal");

            // Fixed seed so this is exactly reproducible run to run.
            ActionPolicy policy = new ActionPolicy(base, new Random(42), "output/summary.csv", "The mean post-error burden is 3.7.");

            PolicyDecision decision = policy.choose(userTask, emptyHistory, signal, true);

            entropies[i] = decision.entropy();

            System.out.printf(
                    "%-12s entropy = %.4f nats%n",
                    state, decision.entropy()
            );
        }

        System.out.println("=".repeat(70));

        double releasedEntropy = entropies[4];
        double normalEntropy = entropies[0];

        boolean allNonNegative = true;
        for (double e : entropies) {
            if (e < 0.0) {
                allNonNegative = false;
            }
        }

        System.out.println();
        System.out.println("Sanity checks:");
        System.out.println("  All entropies >= 0: " + (allNonNegative ? "PASS" : "FAIL -- this would indicate a real bug"));
        System.out.println("  RELEASED entropy < NORMAL entropy: "
                + (releasedEntropy < normalEntropy ? "PASS" : "FAIL -- check ActionPolicy.modulate()'s RELEASED multipliers"));
        System.out.println();
        System.out.println("If both say PASS, the entropy computation is behaving as designed.");
        System.out.println("This does not by itself validate the FULL pipeline (StochasticToolAgent's");
        System.out.println("capture, RunStochasticPolicyExperiment's mean/chi aggregation, or CSV");
        System.out.println("column wiring) -- for that, run a small real sweep and inspect the CSV");
        System.out.println("columns directly (see chat for how to do a reduced-seed pass).");
    }
}
