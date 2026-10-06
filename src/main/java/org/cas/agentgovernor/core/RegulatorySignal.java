/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package org.cas.agentgovernor.core;

/**
 *
 * @author veronique
 */
public class RegulatorySignal {

    private final RegulatoryState state;
    private final int burden;
    private final String instruction;

    public RegulatorySignal(RegulatoryState state, int burden, String instruction) {
        this.state = state;
        this.burden = burden;
        this.instruction = instruction == null ? "" : instruction;
    }

    public RegulatoryState state() {
        return state;
    }

    public int burden() {
        return burden;
    }

    public String instruction() {
        return instruction;
    }

    public boolean isRecoveryMode() {
        return state == RegulatoryState.RECOVERING || state == RegulatoryState.STUCK;
    }

    public boolean isReleased() {
        return state == RegulatoryState.RELEASED;
    }

    @Override
    public String toString() {
        return "RegulatorySignal{"
                + "state=" + state
                + ", burden=" + burden
                + ", instruction='" + instruction + '\''
                + '}';
    }
}
