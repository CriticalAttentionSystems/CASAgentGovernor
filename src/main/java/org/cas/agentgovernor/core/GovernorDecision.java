/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package org.cas.agentgovernor.core;

/**
 *
 * @author veronique
 */

public class GovernorDecision {

    public enum DecisionType {
        ALLOW,
        WARN,
        BLOCK
    }

    private final DecisionType type;
    private final String reason;
    private final RegulatorySignal signal;

    private GovernorDecision(
            DecisionType type,
            String reason,
            RegulatorySignal signal
    ) {
        this.type = type;
        this.reason = reason == null ? "" : reason;
        this.signal = signal;
    }

    public static GovernorDecision allow(String reason, RegulatorySignal signal) {
        return new GovernorDecision(DecisionType.ALLOW, reason, signal);
    }

    public static GovernorDecision warn(String reason, RegulatorySignal signal) {
        return new GovernorDecision(DecisionType.WARN, reason, signal);
    }

    public static GovernorDecision block(String reason, RegulatorySignal signal) {
        return new GovernorDecision(DecisionType.BLOCK, reason, signal);
    }

    public boolean allowed() {
        return type == DecisionType.ALLOW || type == DecisionType.WARN;
    }

    public boolean blocked() {
        return type == DecisionType.BLOCK;
    }

    public boolean warning() {
        return type == DecisionType.WARN;
    }

    public DecisionType type() {
        return type;
    }

    public String reason() {
        return reason;
    }

    public RegulatorySignal signal() {
        return signal;
    }

    @Override
    public String toString() {
        return "GovernorDecision{"
                + "type=" + type
                + ", reason='" + reason + '\''
                + ", signal=" + signal
                + '}';
    }
}