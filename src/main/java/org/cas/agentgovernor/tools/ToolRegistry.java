/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package org.cas.agentgovernor.tools;

import java.util.HashMap;
import java.util.Map;

import org.cas.agentgovernor.core.ToolAction;
import org.cas.agentgovernor.core.ToolResult;

/**
 *
 * @author veronique
 */

public class ToolRegistry {

    private final Map<String, AgentTool> tools = new HashMap<>();

    public void register(AgentTool tool) {
        if (tool == null || tool.name() == null || tool.name().isBlank()) {
            throw new IllegalArgumentException("Cannot register null or unnamed tool.");
        }

        tools.put(tool.name(), tool);
    }

    public ToolResult execute(ToolAction action) {
        if (action == null) {
            return ToolResult.failure("NULL_ACTION", "No tool action was provided.");
        }

        AgentTool tool = tools.get(action.tool());

        if (tool == null) {
            return ToolResult.failure(
                    "UNKNOWN_TOOL",
                    "No tool is registered with name: " + action.tool()
            );
        }

        return tool.execute(action);
    }
}