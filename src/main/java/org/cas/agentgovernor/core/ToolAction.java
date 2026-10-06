/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package org.cas.agentgovernor.core;

import java.util.Objects;

/**
 *
 * @author veronique
 */

public class ToolAction {

    private final String tool;
    private final String actionType;
    private final String target;
    private final String payload;

    public ToolAction(String tool, String actionType, String target, String payload) {
        this.tool = tool == null ? "" : tool;
        this.actionType = actionType == null ? "" : actionType;
        this.target = target == null ? "" : target;
        this.payload = payload == null ? "" : payload;
    }

    public String tool() {
        return tool;
    }

    public String actionType() {
        return actionType;
    }

    public String target() {
        return target;
    }

    public String payload() {
        return payload;
    }

    public boolean sameToolTargetAndAction(ToolAction other) {
        if (other == null) return false;

        return Objects.equals(tool, other.tool)
                && Objects.equals(actionType, other.actionType)
                && Objects.equals(target, other.target);
    }

    @Override
    public String toString() {
        return "ToolAction{"
                + "tool='" + tool + '\''
                + ", actionType='" + actionType + '\''
                + ", target='" + target + '\''
                + ", payload='" + payload + '\''
                + '}';
    }
}