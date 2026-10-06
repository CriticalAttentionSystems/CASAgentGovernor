/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package org.cas.agentgovernor.metrics;

import java.util.HashSet;
import java.util.List;
import java.util.Set;

import org.cas.agentgovernor.core.AgentEvent;
import org.cas.agentgovernor.core.ToolAction;
/**
 *
 * @author veronique
 */


public class AgentRunMetrics {

    private final String agentName;
    private final int totalToolCalls;
    private final int failedToolCalls;
    private final int successfulToolCalls;
    private final int repeatedFailedActions;
    private final int repeatedSuccessfulActions;
    private final int governorInterventions;
    private final int stepsAfterFirstFailure;
    private final int postErrorToolBurden;

    private final int regulatorySignals;
    private final int disturbedSignals;
    private final int recoveringSignals;
    private final int stuckSignals;
    private final int releasedSignals;

    /** True if the run produced any final answer other than the step-budget timeout message. */
    private final boolean answered;

    /**
     * True only if a final answer was given AND it matches the task's expected
     * answer. This is the correctness signal; use this (via {@link #success()})
     * for success-rate reporting, not {@link #answered()}.
     */
    private final boolean correct;

    public AgentRunMetrics(
        String agentName,
        int totalToolCalls,
        int failedToolCalls,
        int successfulToolCalls,
        int repeatedFailedActions,
        int repeatedSuccessfulActions,
        int governorInterventions,
        int regulatorySignals,
        int disturbedSignals,
        int recoveringSignals,
        int stuckSignals,
        int releasedSignals,
        boolean answered,
        boolean correct,
        int stepsAfterFirstFailure,
        int postErrorToolBurden
    ) {
        this.agentName = agentName;
        this.totalToolCalls = totalToolCalls;
        this.failedToolCalls = failedToolCalls;
        this.successfulToolCalls = successfulToolCalls;
        this.repeatedFailedActions = repeatedFailedActions;
        this.repeatedSuccessfulActions = repeatedSuccessfulActions;
        this.governorInterventions = governorInterventions;
        this.answered = answered;
        this.correct = correct;
        this.stepsAfterFirstFailure = stepsAfterFirstFailure;
        this.postErrorToolBurden = postErrorToolBurden;
        this.regulatorySignals = regulatorySignals;
        this.disturbedSignals = disturbedSignals;
        this.recoveringSignals = recoveringSignals;
        this.stuckSignals = stuckSignals;
        this.releasedSignals = releasedSignals;
    }

    /**
     * @param expectedAnswer the task's known-correct answer, or {@code null}/blank if
     *                        no oracle is available (e.g. judging a live LLM's free-text
     *                        answer with no ground truth wired up). When {@code null},
     *                        {@link #success()} / {@link #correct()} will always be false;
     *                        use {@link #answered()} instead for "did it stop and answer".
     */
    public static AgentRunMetrics fromHistory(String agentName, List<AgentEvent> history, String expectedAnswer) {
        int totalToolCalls = 0;
        int failedToolCalls = 0;
        int successfulToolCalls = 0;
        int repeatedFailedActions = 0;
        int repeatedSuccessfulActions = 0;
        int governorInterventions = 0;
        int regulatorySignals = 0;
        int disturbedSignals = 0;
        int recoveringSignals = 0;
        int stuckSignals = 0;
        int releasedSignals = 0;
        boolean answered = false;
        boolean correct = false;

        int firstFailureIndex = -1;
        int toolCallsAfterFirstFailure = 0;

        Set<String> failedActionKeys = new HashSet<>();
        Set<String> successfulActionKeys = new HashSet<>();

        for (int i = 0; i < history.size(); i++) {
            AgentEvent event = history.get(i);

            switch (event.eventType()) {
                case GOVERNOR_INTERVENTION -> governorInterventions++;

                case FINAL_ANSWER -> {
                    String message = event.message();

                    if (message != null && !message.isBlank()
                            && !message.toLowerCase().contains("stopped")) {
                        answered = true;

                        if (matchesExpected(message, expectedAnswer)) {
                            correct = true;
                        }
                    }
                }
                case REGULATORY_SIGNAL -> {
                    regulatorySignals++;

                    if (event.regulatorySignal() != null) {
                        switch (event.regulatorySignal().state()) {
                            case DISTURBED -> disturbedSignals++;
                            case RECOVERING -> recoveringSignals++;
                            case STUCK -> stuckSignals++;
                            case RELEASED -> releasedSignals++;
                            default -> {
                                // NORMAL
                            }
                        }
                    }
                }
                case TOOL_CALL -> {
                    totalToolCalls++;

                    if (firstFailureIndex >= 0) {
                        toolCallsAfterFirstFailure++;
                    }

                    ToolAction action = event.action();
                    String key = actionKey(action);

                    if (event.isFailedToolCall()) {
                        failedToolCalls++;

                        if (firstFailureIndex < 0) {
                            firstFailureIndex = i;
                        }

                        if (failedActionKeys.contains(key)) {
                            repeatedFailedActions++;
                        } else {
                            failedActionKeys.add(key);
                        }

                    } else if (event.isSuccessfulToolCall()) {
                        successfulToolCalls++;

                        if (successfulActionKeys.contains(key)) {
                            repeatedSuccessfulActions++;
                        } else {
                            successfulActionKeys.add(key);
                        }
                    }
                }

                default -> {
                    // Ignore notes.
                }
            }
        }

        int stepsAfterFirstFailure = toolCallsAfterFirstFailure;

        return new AgentRunMetrics(
                agentName,
                totalToolCalls,
                failedToolCalls,
                successfulToolCalls,
                repeatedFailedActions,
                repeatedSuccessfulActions,
                governorInterventions,
                regulatorySignals,
                disturbedSignals,
                recoveringSignals,
                stuckSignals,
                releasedSignals,
                answered,
                correct,
                stepsAfterFirstFailure,
                toolCallsAfterFirstFailure
        );
    }

    /**
     * Correctness check between a produced final answer and the task's expected
     * answer. Exact match (after trimming/whitespace-collapsing/case-folding)
     * covers the deterministic mock LLMs, which echo expectedAnswer verbatim.
     * The containment fallback gives free-text backends (e.g. a live LLM) partial
     * credit when they restate the expected content verbatim inside a longer
     * answer; it is a strict proxy and will under-count real paraphrased answers,
     * so treat it as a lower bound on true correctness, not a full grader.
     */
    private static boolean matchesExpected(String actual, String expectedAnswer) {
        if (expectedAnswer == null || expectedAnswer.isBlank()) {
            return false;
        }

        String normalizedActual = normalizeForComparison(actual);
        String normalizedExpected = normalizeForComparison(expectedAnswer);

        if (normalizedExpected.isEmpty()) {
            return false;
        }

        return normalizedActual.equals(normalizedExpected)
                || normalizedActual.contains(normalizedExpected);
    }

    private static String normalizeForComparison(String text) {
        return text == null ? "" : text.trim().toLowerCase().replaceAll("\\s+", " ");
    }

    private static String actionKey(ToolAction action) {
        if (action == null) {
            return "NULL_ACTION";
        }

        return action.tool() + "::" + action.actionType() + "::" + action.target();
    }

    public String toCsvHeader() {
        return "agent,total_tool_calls,failed_tool_calls,successful_tool_calls,"
                + "repeated_failed_actions,repeated_successful_actions,"
                + "governor_interventions,regulatory_signals,disturbed_signals,"
                + "recovering_signals,stuck_signals,released_signals,"
                + "answered,success,steps_after_first_failure,post_error_tool_burden";
    }

    public String toCsvRow() {
        return String.join(",",
                agentName,
                Integer.toString(totalToolCalls),
                Integer.toString(failedToolCalls),
                Integer.toString(successfulToolCalls),
                Integer.toString(repeatedFailedActions),
                Integer.toString(repeatedSuccessfulActions),
                Integer.toString(governorInterventions),
                Integer.toString(regulatorySignals),
                Integer.toString(disturbedSignals),
                Integer.toString(recoveringSignals),
                Integer.toString(stuckSignals),
                Integer.toString(releasedSignals),
                Boolean.toString(answered),
                Boolean.toString(correct),
                Integer.toString(stepsAfterFirstFailure),
                Integer.toString(postErrorToolBurden)
        );
    }

    @Override
    public String toString() {
        return """
                AgentRunMetrics{
                  agent='%s',
                  totalToolCalls=%d,
                  failedToolCalls=%d,
                  successfulToolCalls=%d,
                  repeatedFailedActions=%d,
                  repeatedSuccessfulActions=%d,
                  governorInterventions=%d,
                  regulatorySignals=%d,
                  disturbedSignals=%d,
                  recoveringSignals=%d,
                  stuckSignals=%d,
                  releasedSignals=%d,
                  answered=%s,
                  success(correct)=%s,
                  stepsAfterFirstFailure=%d,
                  postErrorToolBurden=%d
                }
                """.formatted(
                agentName,
                totalToolCalls,
                failedToolCalls,
                successfulToolCalls,
                repeatedFailedActions,
                repeatedSuccessfulActions,
                governorInterventions,
                regulatorySignals,
                disturbedSignals,
                recoveringSignals,
                stuckSignals,
                releasedSignals,
                answered,
                correct,
                stepsAfterFirstFailure,
                postErrorToolBurden
        );
    }
    
    public String agentName() {
        return agentName;
    }

    public int totalToolCalls() {
        return totalToolCalls;
    }

    public int failedToolCalls() {
        return failedToolCalls;
    }

    public int successfulToolCalls() {
        return successfulToolCalls;
    }

    public int repeatedFailedActions() {
        return repeatedFailedActions;
    }

    public int repeatedSuccessfulActions() {
        return repeatedSuccessfulActions;
    }

    public int governorInterventions() {
        return governorInterventions;
    }

    public int regulatorySignals() {
        return regulatorySignals;
    }

    public int disturbedSignals() {
        return disturbedSignals;
    }

    public int recoveringSignals() {
        return recoveringSignals;
    }

    public int stuckSignals() {
        return stuckSignals;
    }

    public int releasedSignals() {
        return releasedSignals;
    }

    /** True only if the final answer matched the task's expected answer (real correctness). */
    public boolean success() {
        return correct;
    }

    /** True only if the final answer matched the task's expected answer (real correctness). */
    public boolean correct() {
        return correct;
    }

    /** True if any final answer was given at all (correct or not) rather than timing out. */
    public boolean answered() {
        return answered;
    }

    public int stepsAfterFirstFailure() {
        return stepsAfterFirstFailure;
    }

    public int postErrorToolBurden() {
        return postErrorToolBurden;
    }
    
    public static String emptyHeaderWithoutAgentPrefix() {
        return "agent,total_tool_calls,failed_tool_calls,successful_tool_calls,"
                + "repeated_failed_actions,repeated_successful_actions,"
                + "governor_interventions,regulatory_signals,disturbed_signals,"
                + "recovering_signals,stuck_signals,released_signals,"
                + "answered,success,steps_after_first_failure,post_error_tool_burden";
    }
}