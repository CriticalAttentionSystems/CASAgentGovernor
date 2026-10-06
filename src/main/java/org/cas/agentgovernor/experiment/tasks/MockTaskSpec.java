/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package org.cas.agentgovernor.experiment.tasks;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;

/**
 *
 * @author veronique
 */

public class MockTaskSpec {

    private final String taskId;
    private final String prompt;
    private final String expectedTarget;
    private final String expectedAnswer;
    private final Map<String, String> files;

    /**
     * Null for every task that only needs one file read (the original,
     * default case). Set only for tasks whose answer requires reading
     * MULTIPLE specific files (e.g. MockTaskSet's H002) -- see
     * CASGovernor.forExpectedTargets() for why this exists and the bug
     * it fixes.
     */
    private final Set<String> requiredTargets;

    /**
     * Null for every task except ones where multiple independent sources
     * each individually suffice (e.g. a primary + reliable backup, see
     * MockTaskSet's H005) -- see CASGovernor.forAnyOfExpectedTargets().
     * Mutually exclusive with requiredTargets in practice: a task needs
     * either "read all of these" or "read any one of these," not both.
     */
    private final Set<String> acceptableTargets;

    /**
     * Empty for every task except ones deliberately testing tool
     * unreliability (see FlakyFileTool and MockTaskSet's H005). Maps a
     * target path to the number of READ_FILE attempts against it that
     * should fail (with a transient-error result) before succeeding.
     */
    private final Map<String, Integer> flakyReadFailures;

    public MockTaskSpec(
            String taskId,
            String prompt,
            String expectedTarget,
            String expectedAnswer,
            Map<String, String> files
    ) {
        this(taskId, prompt, expectedTarget, expectedAnswer, files, null, null);
    }

    public MockTaskSpec(
            String taskId,
            String prompt,
            String expectedTarget,
            String expectedAnswer,
            Map<String, String> files,
            Set<String> requiredTargets
    ) {
        this(taskId, prompt, expectedTarget, expectedAnswer, files, requiredTargets, null);
    }

    /**
     * @param requiredTargets   see the 6-arg constructor / requiredTargets() doc.
     * @param flakyReadFailures see the field doc above. Pass null for the common
     *                          (non-flaky) case.
     */
    public MockTaskSpec(
            String taskId,
            String prompt,
            String expectedTarget,
            String expectedAnswer,
            Map<String, String> files,
            Set<String> requiredTargets,
            Map<String, Integer> flakyReadFailures
    ) {
        this(taskId, prompt, expectedTarget, expectedAnswer, files, requiredTargets, null, flakyReadFailures);
    }

    /**
     * Full constructor. See the individual field docs above for
     * requiredTargets (AND), acceptableTargets (OR), and flakyReadFailures.
     * Pass null for any that don't apply -- every existing task does, via
     * the shorter constructor overloads above.
     */
    public MockTaskSpec(
            String taskId,
            String prompt,
            String expectedTarget,
            String expectedAnswer,
            Map<String, String> files,
            Set<String> requiredTargets,
            Set<String> acceptableTargets,
            Map<String, Integer> flakyReadFailures
    ) {
        this.taskId = taskId;
        this.prompt = prompt;
        this.expectedTarget = expectedTarget;
        this.expectedAnswer = expectedAnswer;
        this.files = files == null ? new LinkedHashMap<>() : new LinkedHashMap<>(files);
        this.requiredTargets = (requiredTargets == null || requiredTargets.isEmpty())
                ? null
                : new java.util.LinkedHashSet<>(requiredTargets);
        this.acceptableTargets = (acceptableTargets == null || acceptableTargets.isEmpty())
                ? null
                : new java.util.LinkedHashSet<>(acceptableTargets);
        this.flakyReadFailures = (flakyReadFailures == null || flakyReadFailures.isEmpty())
                ? Map.of()
                : new LinkedHashMap<>(flakyReadFailures);
    }

    public String taskId() {
        return taskId;
    }

    public String prompt() {
        return prompt;
    }

    public String expectedTarget() {
        return expectedTarget;
    }

    public String expectedAnswer() {
        return expectedAnswer;
    }

    public Map<String, String> files() {
        return files;
    }

    /**
     * The full set of files that must be read for this task to be
     * answerable. Falls back to the single {@code expectedTarget} when no
     * explicit multi-target set was provided -- so every pre-existing task
     * behaves exactly as before.
     */
    public Set<String> requiredTargets() {
        return requiredTargets == null ? Set.of(expectedTarget) : requiredTargets;
    }

    /** True if this task genuinely requires more than one file to be read. */
    public boolean isMultiTarget() {
        return requiredTargets != null && requiredTargets.size() > 1;
    }

    /**
     * Files where reading ANY ONE suffices (e.g. a primary source plus a
     * reliable backup). Falls back to the single {@code expectedTarget}
     * when no explicit set was provided.
     */
    public Set<String> acceptableTargets() {
        return acceptableTargets == null ? Set.of(expectedTarget) : acceptableTargets;
    }

    /** True if this task has more than one independently-sufficient source. */
    public boolean hasAcceptableTargets() {
        return acceptableTargets != null && acceptableTargets.size() > 1;
    }

    /** Target path -> number of READ_FILE attempts that fail before succeeding. Empty for non-flaky tasks. */
    public Map<String, Integer> flakyReadFailures() {
        return flakyReadFailures;
    }

    /** True if this task uses FlakyFileTool-style deliberate tool unreliability. */
    public boolean hasFlakyReads() {
        return !flakyReadFailures.isEmpty();
    }
}