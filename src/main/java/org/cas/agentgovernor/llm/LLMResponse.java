/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package org.cas.agentgovernor.llm;

import org.cas.agentgovernor.core.ToolAction;

/**
 *
 * @author veronique
 */

public class LLMResponse {

    private final ToolAction proposedAction;
    private final boolean wantsFinalAnswer;
    private final String finalAnswer;
    private final String rawText;

    public LLMResponse(
            ToolAction proposedAction,
            boolean wantsFinalAnswer,
            String finalAnswer,
            String rawText
    ) {
        this.proposedAction = proposedAction;
        this.wantsFinalAnswer = wantsFinalAnswer;
        this.finalAnswer = finalAnswer == null ? "" : finalAnswer;
        this.rawText = rawText == null ? "" : rawText;
    }

    public ToolAction proposedAction() {
        return proposedAction;
    }

    public boolean wantsFinalAnswer() {
        return wantsFinalAnswer;
    }

    public String finalAnswer() {
        return finalAnswer;
    }

    public String rawText() {
        return rawText;
    }

    @Override
    public String toString() {
        if (wantsFinalAnswer) {
            return "LLMResponse{FINAL_ANSWER: " + finalAnswer + "}";
        }
        return "LLMResponse{action=" + proposedAction + "}";
    }
}
