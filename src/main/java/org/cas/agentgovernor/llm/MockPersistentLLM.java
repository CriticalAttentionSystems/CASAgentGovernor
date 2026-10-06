/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package org.cas.agentgovernor.llm;

import org.cas.agentgovernor.core.AgentEvent;
import org.cas.agentgovernor.core.RegulatoryState;
import org.cas.agentgovernor.core.ToolAction;
import org.cas.agentgovernor.core.ToolDiscovery;

/**
 *
 * @author veronique
 */

/**
 * A more persistent mock LLM.
 *
 * Ungoverned behavior:
 *   Repeats the same failed guessed read several times before recovery.
 *
 * Governed behavior:
 *   Responds to CAS regulatory state modulation and switches to recovery earlier.
 */
public class MockPersistentLLM implements LLMClient {

    private final String expectedTarget;
    private final String expectedAnswer;
    private final int ungovernedRepeatLimit;

    public MockPersistentLLM(String expectedTarget, String expectedAnswer) {
        this(expectedTarget, expectedAnswer, 4);
    }

    public MockPersistentLLM(
            String expectedTarget,
            String expectedAnswer,
            int ungovernedRepeatLimit
    ) {
        this.expectedTarget = expectedTarget;
        this.expectedAnswer = expectedAnswer;
        this.ungovernedRepeatLimit = ungovernedRepeatLimit;
    }

    public String expectedTarget() {
        return expectedTarget;
    }

    public String expectedAnswer() {
        return expectedAnswer;
    }

    @Override
    public LLMResponse generate(LLMRequest request) {

        if (request.regulatorySignal().isReleased()) {
            return new LLMResponse(
                    null,
                    true,
                    expectedAnswer,
                    "MockPersistentLLM final answer under RELEASED state."
            );
        }

        if (hasSuccessfulExpectedRead(request)) {
            return new LLMResponse(
                    null,
                    true,
                    expectedAnswer,
                    "MockPersistentLLM final answer after successful expected read."
            );
        }

        /*
         * This is the important difference from the ungoverned case.
         * If the governor marks the episode as disturbed or recovering,
         * the mock stops repeating the failed action and switches behavior.
         */
        if (request.regulatorySignal().state() == RegulatoryState.DISTURBED
                || request.regulatorySignal().isRecoveryMode()) {
            return recoveryAction(request);
        }

        int toolCalls = countToolCalls(request);

        /*
         * In the absence of a regulatory signal, persist with the same
         * guessed action for several tool calls.
         */
        if (toolCalls < ungovernedRepeatLimit) {
            return new LLMResponse(
                    guessedFirstReadAction(request),
                    false,
                    "",
                    "MockPersistentLLM: persisting with guessed path."
            );
        }

        return recoveryAction(request);
    }

    private ToolAction guessedFirstReadAction(LLMRequest request) {
        String prompt = request.userTask().toLowerCase();

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

    /**
     * Recovery behavior after a disturbance: list the root if not yet
     * listed, then list the next directory actually discovered via prior
     * listings (never guessed from expectedTarget), then read the
     * best-matching discovered-but-unread file. If discovery is fully
     * exhausted (nothing left to list, nothing left to read) without a
     * match, gives up honestly with a generic wrong answer rather than
     * falling back to the harness's known-correct expectedTarget.
     */
    private LLMResponse recoveryAction(LLMRequest request) {
        if (!hasListed(request, ".")) {
            return new LLMResponse(
                    new ToolAction("file", "LIST_DIRECTORY", ".", ""),
                    false,
                    "",
                    "MockPersistentLLM recovery: inspect root workspace."
            );
        }

        String nextDirectory = ToolDiscovery.nextDiscoveredUnlistedDirectory(request.history());

        if (!nextDirectory.isBlank()) {
            return new LLMResponse(
                    new ToolAction("file", "LIST_DIRECTORY", nextDirectory, ""),
                    false,
                    "",
                    "MockPersistentLLM recovery: inspect " + nextDirectory + "."
            );
        }

        String discoveredTarget = ToolDiscovery.discoveredReadCandidateTarget(request.userTask(), request.history());

        if (discoveredTarget != null) {
            return new LLMResponse(
                    new ToolAction("file", "READ_FILE", discoveredTarget, ""),
                    false,
                    "",
                    "MockPersistentLLM recovery: read discovered candidate " + discoveredTarget + "."
            );
        }

        return new LLMResponse(
                null,
                true,
                "Unable to determine the answer from the available information.",
                "MockPersistentLLM recovery: exhausted every directory and file discovered via listing "
                        + "without finding a match; giving up rather than guessing the answer."
        );
    }

    private boolean hasListed(LLMRequest request, String target) {
        String normalizedTarget = normalizeTarget(target);

        for (AgentEvent event : request.history()) {
            if (!event.isSuccessfulToolCall()) {
                continue;
            }

            if (event.action() == null) {
                continue;
            }

            boolean isList =
                    "file".equalsIgnoreCase(event.action().tool())
                            && "LIST_DIRECTORY".equalsIgnoreCase(event.action().actionType());

            if (!isList) {
                continue;
            }

            if (normalizeTarget(event.action().target()).equals(normalizedTarget)) {
                return true;
            }
        }

        return false;
    }

    private boolean hasSuccessfulExpectedRead(LLMRequest request) {
        for (AgentEvent event : request.history()) {
            if (!event.isSuccessfulToolCall()) {
                continue;
            }

            if (event.action() == null) {
                continue;
            }

            boolean isRead =
                    "file".equalsIgnoreCase(event.action().tool())
                            && "READ_FILE".equalsIgnoreCase(event.action().actionType());

            boolean isExpected =
                    normalizeTarget(event.action().target()).equals(normalizeTarget(expectedTarget));

            if (isRead && isExpected) {
                return true;
            }
        }

        return false;
    }

    private int countToolCalls(LLMRequest request) {
        int count = 0;

        for (AgentEvent event : request.history()) {
            if (event.isToolCall()) {
                count++;
            }
        }

        return count;
    }

    private String normalizeTarget(String target) {
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
