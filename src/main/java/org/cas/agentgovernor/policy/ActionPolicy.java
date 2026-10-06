/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package org.cas.agentgovernor.policy;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.cas.agentgovernor.core.AgentEvent;
import org.cas.agentgovernor.core.RegulatorySignal;
import org.cas.agentgovernor.core.RegulatoryState;
import org.cas.agentgovernor.core.ToolAction;
import org.cas.agentgovernor.core.ToolDiscovery;

/**
 *
 * @author veronique
 */

public class ActionPolicy {

    public static class PolicyDecision {
        private final ToolAction action;
        private final boolean wantsFinalAnswer;
        private final String finalAnswer;
        private final double entropy;

        private PolicyDecision(ToolAction action, boolean wantsFinalAnswer, String finalAnswer, double entropy) {
            this.action = action;
            this.wantsFinalAnswer = wantsFinalAnswer;
            this.finalAnswer = finalAnswer == null ? "" : finalAnswer;
            this.entropy = entropy;
        }

        public static PolicyDecision action(ToolAction action) {
            return new PolicyDecision(action, false, "", 0.0);
        }

        public static PolicyDecision finalAnswer(String answer) {
            return new PolicyDecision(null, true, answer, 0.0);
        }

        /** Returns a copy of this decision with the given entropy attached. Used only by sample(). */
        private PolicyDecision withEntropy(double entropy) {
            return new PolicyDecision(this.action, this.wantsFinalAnswer, this.finalAnswer, entropy);
        }

        public ToolAction action() {
            return action;
        }

        public boolean wantsFinalAnswer() {
            return wantsFinalAnswer;
        }

        public String finalAnswer() {
            return finalAnswer;
        }

        /**
         * Shannon entropy, in nats (natural log), of the weighted-choice
         * distribution this decision was sampled from -- i.e. the entropy
         * of the categorical distribution actually formed by normalizing
         * every candidate action's weight into a probability at this step.
         * This is a literal, real S_vN-analogue (see
         * CognitiveStateMetrics.vonNeumannEntropy in the companion
         * IRAM-Omega-Q continuous-formalism project): the discrete policy
         * genuinely IS a probability distribution over candidate next
         * actions at each step, computed the same way, not a metaphor for
         * one. 0.0 for deterministic decisions (e.g. the
         * hasSuccessfulExpectedRead shortcut in choose(), which has no
         * distribution to sample from at all -- one outcome with
         * probability 1 has zero entropy, which is the mathematically
         * correct value, not a placeholder).
         */
        public double entropy() {
            return entropy;
        }
    }

    private static class WeightedChoice {
        private final double weight;
        private final PolicyDecision decision;

        private WeightedChoice(double weight, PolicyDecision decision) {
            this.weight = Math.max(0.0, weight);
            this.decision = decision;
        }
    }

    private final PolicyParameters baseParameters;
    private final Random random;
    private final String expectedTarget;
    private final String expectedAnswer;

    public ActionPolicy(
            PolicyParameters baseParameters,
            Random random,
            String expectedTarget,
            String expectedAnswer
    ) {
        this.baseParameters = baseParameters;
        this.random = random;
        this.expectedTarget = expectedTarget;
        this.expectedAnswer = expectedAnswer;
    }

    public PolicyDecision choose(
            String userTask,
            List<AgentEvent> history,
            RegulatorySignal signal,
            boolean applyRegulationWeights
    ) {
        if (hasSuccessfulExpectedRead(history)) {
            return PolicyDecision.finalAnswer(expectedAnswer);
        }

        PolicyParameters p = applyRegulationWeights
                ? modulate(baseParameters, signal)
                : baseParameters;

        List<WeightedChoice> choices = new ArrayList<>();

        ToolAction lastFailed = lastFailedAction(history);
        ToolAction guessed = guessedFirstReadAction(userTask);

        if (lastFailed != null) {
            choices.add(new WeightedChoice(
                    p.retryFailedWeight(),
                    PolicyDecision.action(lastFailed)
            ));
        } else {
            choices.add(new WeightedChoice(
                    p.retryFailedWeight(),
                    PolicyDecision.action(guessed)
            ));
        }

        choices.add(new WeightedChoice(
                p.inspectRootWeight(),
                PolicyDecision.action(new ToolAction("file", "LIST_DIRECTORY", ".", ""))
        ));

        String localDirectory = localDirectoryFromLastFailed(history);
        if (!localDirectory.isBlank()) {
            choices.add(new WeightedChoice(
                    p.inspectLocalWeight(),
                    PolicyDecision.action(new ToolAction("file", "LIST_DIRECTORY", localDirectory, ""))
            ));
        }

        String nextCandidateDirectory = nextDiscoveredUnlistedDirectory(history);
        if (!nextCandidateDirectory.isBlank()) {
            choices.add(new WeightedChoice(
                    p.inspectCandidateDirectoryWeight(),
                    PolicyDecision.action(new ToolAction("file", "LIST_DIRECTORY", nextCandidateDirectory, ""))
            ));
        }

        ToolAction readCandidate = discoveredReadCandidate(userTask, history);
        if (readCandidate != null) {
            choices.add(new WeightedChoice(
                    p.readCandidateWeight(),
                    PolicyDecision.action(readCandidate)
            ));
        }

        choices.add(new WeightedChoice(
                p.randomGuessWeight(),
                PolicyDecision.action(randomGuess(userTask))
        ));

        choices.add(new WeightedChoice(
                p.prematureAnswerWeight(),
                PolicyDecision.finalAnswer("Unable to determine the answer from the available information.")
        ));

        return sample(choices);
    }

    /**
     * Applies the CAS regulatory-state weight multipliers to a base
     * PolicyParameters. Exposed as a public static utility (rather than
     * only invoked live, per-step, inside {@link #choose}) so experiment
     * harnesses can precompute a fixed "regulation-shaped" weight profile
     * once — e.g. to build a static-good baseline agent that uses good
     * weights for the whole episode without ever consulting the governor
     * dynamically. This lets us separate "does dynamic state-conditioned
     * adaptation help" from "do these particular weights help."
     */
    public static PolicyParameters modulate(PolicyParameters p, RegulatorySignal signal) {
        if (signal == null) {
            return p;
        }

        RegulatoryState state = signal.state();

        if (state == RegulatoryState.NORMAL) {
            return p;
        }

        if (state == RegulatoryState.DISTURBED) {
            return p
                    .withRetryMultiplier(0.25)
                    .withInspectMultiplier(2.50)
                    .withPrematureAnswerMultiplier(0.25);
        }

        if (state == RegulatoryState.RECOVERING) {
            return p
                    .withRetryMultiplier(0.15)
                    .withInspectMultiplier(1.50)
                    .withReadCandidateMultiplier(2.50)
                    .withPrematureAnswerMultiplier(0.40);
        }

        if (state == RegulatoryState.STUCK) {
            return p
                    .withRetryMultiplier(0.05)
                    .withInspectMultiplier(3.00)
                    .withReadCandidateMultiplier(1.50)
                    .withPrematureAnswerMultiplier(0.10);
        }

        if (state == RegulatoryState.RELEASED) {
            return p
                    .withRetryMultiplier(0.01)
                    .withInspectMultiplier(0.10)
                    .withReadCandidateMultiplier(0.10)
                    .withPrematureAnswerMultiplier(10.00);
        }

        return p;
    }

    private PolicyDecision sample(List<WeightedChoice> choices) {
        double total = 0.0;

        for (WeightedChoice choice : choices) {
            total += choice.weight;
        }

        double entropy = shannonEntropy(choices, total);

        if (total <= 0.0) {
            return PolicyDecision.action(new ToolAction("file", "LIST_DIRECTORY", ".", "")).withEntropy(entropy);
        }

        double r = random.nextDouble() * total;
        double running = 0.0;

        for (WeightedChoice choice : choices) {
            running += choice.weight;
            if (r <= running) {
                return choice.decision.withEntropy(entropy);
            }
        }

        return choices.get(choices.size() - 1).decision.withEntropy(entropy);
    }

    /**
     * Shannon entropy (nats) of the categorical distribution formed by
     * normalizing each choice's weight into a probability -- exactly the
     * distribution this method's caller (sample()) draws from, computed
     * the same way, not approximated.
     */
    private double shannonEntropy(List<WeightedChoice> choices, double total) {
        if (total <= 0.0) {
            return 0.0;
        }

        double entropy = 0.0;

        for (WeightedChoice choice : choices) {
            if (choice.weight <= 0.0) {
                continue;
            }

            double prob = choice.weight / total;
            entropy -= prob * Math.log(prob);
        }

        return entropy;
    }

    private ToolAction guessedFirstReadAction(String userTask) {
        String prompt = userTask == null ? "" : userTask.toLowerCase();

        if (prompt.contains("examples/trial_config.json")) {
            return new ToolAction("file", "READ_FILE", "examples/trial_config.json", "");
        }

        if (prompt.contains("configs/session_config.json")) {
            return new ToolAction("file", "READ_FILE", "configs/session_config.json", "");
        }

        if (prompt.contains("examples/run_summary.csv")) {
            return new ToolAction("file", "READ_FILE", "examples/run_summary.csv", "");
        }

        if (prompt.contains("output/summary.csv")) {
            return new ToolAction("file", "READ_FILE", "output/summary.csv", "");
        }

        if (prompt.contains("metrics.csv")) {
            return new ToolAction("file", "READ_FILE", "metrics.csv", "");
        }

        if (prompt.contains("latest summary")) {
            return new ToolAction("file", "READ_FILE", "summary.csv", "");
        }

        if (prompt.contains("config.json")) {
            return new ToolAction("file", "READ_FILE", "config.json", "");
        }

        return new ToolAction("file", "READ_FILE", "examples/trial_config.json", "");
    }

    private ToolAction randomGuess(String userTask) {
        int k = random.nextInt(6);

        return switch (k) {
            case 0 -> new ToolAction("file", "READ_FILE", "config.json", "");
            case 1 -> new ToolAction("file", "READ_FILE", "summary.csv", "");
            case 2 -> new ToolAction("file", "READ_FILE", "metrics.csv", "");
            case 3 -> new ToolAction("file", "LIST_DIRECTORY", "examples", "");
            case 4 -> new ToolAction("file", "LIST_DIRECTORY", "output", "");
            default -> new ToolAction("file", "LIST_DIRECTORY", "configs", "");
        };
    }

    private ToolAction lastFailedAction(List<AgentEvent> history) {
        for (int i = history.size() - 1; i >= 0; i--) {
            AgentEvent event = history.get(i);

            if (event.isFailedToolCall()) {
                return event.action();
            }
        }

        return null;
    }

    private String localDirectoryFromLastFailed(List<AgentEvent> history) {
        ToolAction failed = lastFailedAction(history);

        if (failed == null || failed.target() == null || failed.target().isBlank()) {
            return "";
        }

        String target = normalize(failed.target());
        int slash = target.lastIndexOf('/');

        if (slash <= 0) {
            return ".";
        }

        return target.substring(0, slash);
    }

    /**
     * Picks a subdirectory to inspect next, drawn ONLY from directories the
     * policy has actually seen reported in prior successful LIST_DIRECTORY
     * outputs (breadth-first over the discovered tree), not from splitting
     * the harness's known-correct {@code expectedTarget} path. This means
     * the policy has to actually discover the file's location by listing,
     * the same way a real agent would, rather than beelining down a path
     * it was never shown. Delegates to the shared {@link ToolDiscovery}
     * utility so this stays consistent with the mock LLMs.
     */
    private String nextDiscoveredUnlistedDirectory(List<AgentEvent> history) {
        return ToolDiscovery.nextDiscoveredUnlistedDirectory(history);
    }

    /**
     * Picks a file to read from ones actually reported in prior successful
     * LIST_DIRECTORY outputs, scored by shared keyword-overlap against the
     * task prompt (see {@link ToolDiscovery#discoveredReadCandidateTarget}).
     * Has no access to the harness's expectedTarget: it can only act on
     * files the simulated tool calls actually surfaced, so "recovery" here
     * means real discovery, not privileged knowledge.
     */
    private ToolAction discoveredReadCandidate(String userTask, List<AgentEvent> history) {
        String target = ToolDiscovery.discoveredReadCandidateTarget(userTask, history);
        return target == null ? null : new ToolAction("file", "READ_FILE", target, "");
    }

    private boolean hasSuccessfulExpectedRead(List<AgentEvent> history) {
        for (AgentEvent event : history) {
            if (!event.isSuccessfulToolCall()) {
                continue;
            }

            ToolAction action = event.action();

            if (action == null) {
                continue;
            }

            boolean isRead = "file".equalsIgnoreCase(action.tool())
                    && "READ_FILE".equalsIgnoreCase(action.actionType());

            boolean isExpected = normalize(action.target()).equals(normalize(expectedTarget));

            if (isRead && isExpected) {
                return true;
            }
        }

        return false;
    }

    private String normalize(String target) {
        if (target == null || target.isBlank()) {
            return ".";
        }

        String t = target.trim();

        while (t.endsWith("/") || t.endsWith("\\")) {
            t = t.substring(0, t.length() - 1);
        }

        if (t.isBlank() || "./".equals(t)) {
            return ".";
        }

        return t;
    }
}