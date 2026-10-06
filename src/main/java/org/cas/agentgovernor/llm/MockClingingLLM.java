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



public class MockClingingLLM implements LLMClient {

    private final String expectedTarget;
    private final String expectedAnswer;

    public MockClingingLLM() {
        this(
                "configs/trial_config_demo.json",
                "The trial config defines three phases: baseline from 0 to 120 seconds, disturbance from 120 to 240 seconds, and recovery from 240 to 420 seconds."
        );
    }

    public MockClingingLLM(String expectedTarget, String expectedAnswer) {
        this.expectedTarget = expectedTarget;
        this.expectedAnswer = expectedAnswer;
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
                    "Mock final answer under RELEASED state."
            );
        }

        if (hasSuccessfulExpectedRead(request)) {
            return new LLMResponse(
                    null,
                    true,
                    expectedAnswer,
                    "Mock final answer after successful expected read."
            );
        }

        if (request.regulatorySignal().state() == RegulatoryState.DISTURBED
                || request.regulatorySignal().isRecoveryMode()) {
            return recoveryAction(request);
        }

        int toolCalls = countToolCalls(request);

        if (toolCalls == 0) {
            return new LLMResponse(
                    guessedFirstReadAction(request),
                    false,
                    "",
                    "Mock step 1: trying the guessed path from the prompt."
            );
        }

        if (toolCalls == 1) {
            return new LLMResponse(
                    guessedFirstReadAction(request),
                    false,
                    "",
                    "Mock step 2: clinging to the same failed path."
            );
        }

        return recoveryAction(request);
    }

    private ToolAction guessedFirstReadAction(LLMRequest request) {
        String prompt = request.userTask().toLowerCase();

        // Most specific first: "trial_config.json" contains "config.json",
        // so this must come before the generic config.json check.
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

        // Generic stale/old config reference.
        if (prompt.contains("config.json")) {
            return new ToolAction("file", "READ_FILE", "config.json", "");
        }

        // Default wrong guess for config-style tasks.
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
                    "Mock recovery: inspect root workspace."
            );
        }

        String nextDirectory = ToolDiscovery.nextDiscoveredUnlistedDirectory(request.history());

        if (!nextDirectory.isBlank()) {
            return new LLMResponse(
                    new ToolAction("file", "LIST_DIRECTORY", nextDirectory, ""),
                    false,
                    "",
                    "Mock recovery: inspect " + nextDirectory + "."
            );
        }

        String discoveredTarget = ToolDiscovery.discoveredReadCandidateTarget(request.userTask(), request.history());

        if (discoveredTarget != null) {
            return new LLMResponse(
                    new ToolAction("file", "READ_FILE", discoveredTarget, ""),
                    false,
                    "",
                    "Mock recovery: read discovered candidate " + discoveredTarget + "."
            );
        }

        return new LLMResponse(
                null,
                true,
                "Unable to determine the answer from the available information.",
                "Mock recovery: exhausted every directory and file discovered via listing "
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