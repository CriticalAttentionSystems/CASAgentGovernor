/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package org.cas.agentgovernor.core;

/**
 *
 * @author veronique
 */

public class ToolResult {

    private final boolean success;
    private final String errorType;
    private final String message;
    private final String output;

    public ToolResult(
            boolean success,
            String errorType,
            String message,
            String output
    ) {
        this.success = success;
        this.errorType = errorType == null ? "" : errorType;
        this.message = message == null ? "" : message;
        this.output = output == null ? "" : output;
    }

    public static ToolResult success(String output) {
        return new ToolResult(true, "NONE", "OK", output);
    }

    public static ToolResult failure(String errorType, String message) {
        return new ToolResult(false, errorType, message, "");
    }

    public boolean success() {
        return success;
    }

    public String errorType() {
        return errorType;
    }

    public String message() {
        return message;
    }

    public String output() {
        return output;
    }

    @Override
    public String toString() {
        return "ToolResult{"
                + "success=" + success
                + ", errorType='" + errorType + '\''
                + ", message='" + message + '\''
                + ", output='" + abbreviate(output, 120) + '\''
                + '}';
    }

    private static String abbreviate(String text, int maxLength) {
        if (text == null) return "";
        if (text.length() <= maxLength) return text;
        return text.substring(0, maxLength) + "...";
    }
}