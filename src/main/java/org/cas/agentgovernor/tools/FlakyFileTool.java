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
 * Wraps a real {@link FileTool} and injects controlled, deterministic
 * transient failures on specific READ_FILE targets: the first N attempts
 * to read a configured target fail with a distinguishably TRANSIENT error
 * (not "file not found" -- a message implying flakiness, not a wrong
 * path), and the (N+1)th and every later attempt on that same target
 * delegates to the real underlying read and succeeds normally.
 *
 * <p><b>Why this exists.</b> Every other task in this codebase (the
 * original 8, plus H001-H004) has "wrong guess" failures that are
 * DETERMINISTICALLY wrong: the correct recovery is always to look
 * SOMEWHERE ELSE, never to retry the same action -- which is exactly what
 * CASGovernor's repeat-blocking (wouldRepeatFailedAction) already rewards.
 * That means every task tried so far structurally favors governed's
 * blocking mechanism by construction; none of them could reveal blocking
 * as a LIABILITY. A target that is flaky rather than wrong inverts that:
 * retrying the exact same action is sometimes the objectively correct
 * recovery strategy. Under governed/block_only, wouldRepeatFailedAction()
 * will hard-block any second attempt at the same target regardless of
 * whether retrying would have worked -- so this tool lets a task exist
 * (see MockTaskSet's H005) where governance can genuinely underperform an
 * agent that's simply willing to try again. A reliable backup source
 * exists in that task (undiscoverable from the prompt alone) so a blocked
 * agent still has a fair path to succeed -- this tests a real tradeoff,
 * it isn't rigged to make either condition lose.
 *
 * @author veronique
 */
public class FlakyFileTool implements AgentTool {

    private final FileTool delegate;
    private final Map<String, Integer> failuresBeforeSuccess;
    private final Map<String, Integer> attemptCounts = new HashMap<>();

    /**
     * @param delegate               the real FileTool this wraps for every
     *                                non-flaky action and for the eventual
     *                                successful read once a flaky target's
     *                                failure count is exhausted.
     * @param failuresBeforeSuccess  normalized target path -> number of READ_FILE
     *                              attempts that must fail before the
     *                              (N+1)th attempt on that same target is
     *                              allowed to succeed. A target not present
     *                              in this map is never made flaky --
     *                              behaves exactly like plain FileTool.
     *                              Only applies to READ_FILE; LIST_DIRECTORY
     *                              is never made flaky by this class.
     */
    public FlakyFileTool(FileTool delegate, Map<String, Integer> failuresBeforeSuccess) {
        this.delegate = delegate;
        this.failuresBeforeSuccess = new HashMap<>();
        if (failuresBeforeSuccess != null) {
            for (Map.Entry<String, Integer> entry : failuresBeforeSuccess.entrySet()) {
                this.failuresBeforeSuccess.put(normalize(entry.getKey()), entry.getValue());
            }
        }
    }

    @Override
    public String name() {
        return delegate.name();
    }

    @Override
    public ToolResult execute(ToolAction action) {
        if (action == null || !"READ_FILE".equalsIgnoreCase(action.actionType())) {
            return delegate.execute(action);
        }

        String key = normalize(action.target());
        Integer requiredFailures = failuresBeforeSuccess.get(key);

        if (requiredFailures == null || requiredFailures <= 0) {
            return delegate.execute(action);
        }

        int attemptsSoFar = attemptCounts.merge(key, 1, Integer::sum);

        if (attemptsSoFar <= requiredFailures) {
            return ToolResult.failure(
                    "TRANSIENT_READ_ERROR",
                    "Temporary read error: connection reset while reading " + action.target() + "."
            );
        }

        return delegate.execute(action);
    }

    private String normalize(String target) {
        if (target == null || target.isBlank()) {
            return ".";
        }

        String t = target.trim();

        while (t.endsWith("/") || t.endsWith("\\")) {
            t = t.substring(0, t.length() - 1);
        }

        return t.isBlank() ? "." : t;
    }
}
