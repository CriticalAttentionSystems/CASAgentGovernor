/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package org.cas.agentgovernor.core;

import java.time.Instant;

/**
 *
 * @author veronique
 */

public class AgentEvent {

    public enum EventType {
        TOOL_CALL,
        GOVERNOR_INTERVENTION,
        REGULATORY_SIGNAL,
        FINAL_ANSWER,
        NOTE
    }

    private final RegulatorySignal regulatorySignal;
    private final Instant timestamp;
    private final EventType eventType;
    private final ToolAction action;
    private final ToolResult result;
    private final String message;

    private AgentEvent(
            EventType eventType,
            ToolAction action,
            ToolResult result,
            String message,
            RegulatorySignal regulatorySignal
    ) {
        this.timestamp = Instant.now();
        this.eventType = eventType;
        this.action = action;
        this.result = result;
        this.message = message == null ? "" : message;
        this.regulatorySignal = regulatorySignal;
    }

    public static AgentEvent regulatorySignal(RegulatorySignal signal) {
        return new AgentEvent(
                EventType.REGULATORY_SIGNAL,
                null,
                null,
                signal == null ? "" : signal.instruction(),
                signal
        );
    }
    
    public RegulatorySignal regulatorySignal() {
        return regulatorySignal;
    }

    public boolean isRegulatorySignal() {
        return eventType == EventType.REGULATORY_SIGNAL;
    }

    public static AgentEvent toolCall(ToolAction action, ToolResult result) {
        return new AgentEvent(EventType.TOOL_CALL, action, result, "", null);
    }

    public static AgentEvent governorIntervention(String message) {
        return new AgentEvent(EventType.GOVERNOR_INTERVENTION, null, null, message, null);
    }

    public static AgentEvent finalAnswer(String answer) {
        return new AgentEvent(EventType.FINAL_ANSWER, null, null, answer, null);
    }

    public static AgentEvent note(String message) {
        return new AgentEvent(EventType.NOTE, null, null, message, null);
    }
    
    public Instant timestamp() {
        return timestamp;
    }

    public EventType eventType() {
        return eventType;
    }

    public ToolAction action() {
        return action;
    }

    public ToolResult result() {
        return result;
    }

    public String message() {
        return message;
    }

    public boolean isToolCall() {
        return eventType == EventType.TOOL_CALL;
    }

    public boolean isFailedToolCall() {
        return isToolCall() && result != null && !result.success();
    }

    public boolean isSuccessfulToolCall() {
        return isToolCall() && result != null && result.success();
    }

    public String errorType() {
        if (result == null) return "";
        return result.errorType();
    }

    @Override
    public String toString() {
        if (eventType == EventType.TOOL_CALL) {
            return "TOOL_CALL action=" + action + " result=" + result;
        }

        if (eventType == EventType.GOVERNOR_INTERVENTION) {
            return "GOVERNOR_INTERVENTION message=\"" + message + "\"";
        }

        if (eventType == EventType.REGULATORY_SIGNAL) {
            return "REGULATORY_SIGNAL " + regulatorySignal;
        }

        if (eventType == EventType.FINAL_ANSWER) {
            return "FINAL_ANSWER message=\"" + message + "\"";
        }

        return "NOTE message=\"" + message + "\"";
    }
}