/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package org.cas.agentgovernor.tools;

import org.cas.agentgovernor.core.ToolAction;
import org.cas.agentgovernor.core.ToolResult;

/**
 *
 * @author veronique
 */

public interface AgentTool {

    String name();

    ToolResult execute(ToolAction action);
}