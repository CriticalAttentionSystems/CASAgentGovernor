/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package org.cas.agentgovernor.core;

import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.function.Predicate;

/**
 *
 * @author veronique
 */


public class CASGovernor {

    /**
     * Repetition score at which sustained looping escalates from RECOVERING
     * to STUCK. Deliberately based on repeated-action counts only (see
     * {@link BurdenBreakdown#stuckScore()}), NOT on the reported burden
     * number as a whole: burden also grows from the generic "post-error
     * tool calls" term, which increases with every step after a failure
     * even during a well-executed, non-repeating recovery. Gating STUCK on
     * total burden would misclassify a long-but-effective exploration (many
     * distinct LIST_DIRECTORY/READ_FILE calls, none repeated) as looping.
     * The default of 6 means, e.g., two distinct repeated-failed actions
     * (3 each) trigger it, or one repeated-failed plus one repeated-successful.
     */
    public static final int DEFAULT_STUCK_SCORE_THRESHOLD = 6;

    /**
     * Decides whether a successfully-read file target counts as "useful
     * content" for the purposes of releasing the agent from tool use.
     *
     * Defaults to accepting any non-blank READ_FILE output, which is the
     * only option available to a general-purpose governor that has no
     * ground truth about which file actually answers the task. Callers
     * that DO know the correct target (e.g. benchmark harnesses driven by
     * a known expected file) should supply a stricter filter via
     * {@link #CASGovernor(Predicate)} or {@link #forExpectedTarget(String)}
     * so that RELEASED is not signaled after reading an irrelevant file.
     */
    private final Predicate<String> relevantTargetFilter;

    /**
     * If non-null, activates multi-target mode: RELEASED (and
     * hasRecoveredUsefulContent) only fires once EVERY target in this set
     * has been successfully read at least once, not just any one of them.
     * Only set by {@link #forExpectedTargets(java.util.Collection)}; every
     * other constructor/factory leaves this null and uses
     * relevantTargetFilter as before -- purely additive, zero behavior
     * change for existing single-target callers.
     *
     * Added after a real bug: a task requiring synthesis across two files
     * (see MockTaskSet's H002) was governed with a single-target governor
     * pointed at only one of them. The governor declared "useful content
     * recovered" the moment that ONE file was read, fired RELEASED, and
     * enforceBlocking then prevented the agent from ever reading the
     * second required file -- deterministically producing wrong answers
     * on every governed run of that task (confirmed directly: the model's
     * own final answers cited "the CAS regulatory signal (state=RELEASED)"
     * as the reason it stopped early). This mode exists so multi-file
     * tasks can be governed correctly instead of avoided.
     */
    private final Set<String> requiredTargets;

    private final int stuckScoreThreshold;

    /**
     * If &gt; 0, activates bounded-retry mode: a failed action whose failures
     * were ALL classified as transient (see {@link #transientErrorTypes})
     * may be retried up to this many times before being hard-blocked, instead
     * of being blocked on the very first repeat. 0 (the default for every
     * existing constructor/factory) preserves the original behavior exactly
     * -- block any repeat of a failed action, unconditionally.
     *
     * Added after a real, replicated finding (MockTaskSet's H005): blocking
     * every repeat uniformly is actively harmful when a failure is
     * transient (e.g. a flaky read that would succeed if retried) rather
     * than a genuine dead end. The original blocking behavior is exactly
     * right for deterministic wrong-path failures (file not found, wrong
     * extension, etc.) -- this mode does not change behavior for those; it
     * only gives a bounded second (or Nth) chance specifically to failures
     * whose errorType is in {@link #transientErrorTypes}, and still blocks
     * once that budget is exhausted, so it cannot degrade into an infinite
     * retry loop if the classification turns out to be wrong.
     *
     * Only set via {@link #withBoundedRetry(int, Set)}.
     */
    private final int maxTransientRetries;

    /**
     * Which {@code ToolResult.errorType()} values count as "transient,
     * worth retrying" under bounded-retry mode. Empty/null (the default)
     * means no error type is ever treated as transient, which combined
     * with maxTransientRetries==0 preserves the original behavior exactly.
     */
    private final Set<String> transientErrorTypes;

    public CASGovernor() {
        this(target -> true, DEFAULT_STUCK_SCORE_THRESHOLD, null, 0, null);
    }

    public CASGovernor(Predicate<String> relevantTargetFilter) {
        this(relevantTargetFilter, DEFAULT_STUCK_SCORE_THRESHOLD, null, 0, null);
    }

    public CASGovernor(Predicate<String> relevantTargetFilter, int stuckScoreThreshold) {
        this(relevantTargetFilter, stuckScoreThreshold, null, 0, null);
    }

    private CASGovernor(Predicate<String> relevantTargetFilter, int stuckScoreThreshold, Set<String> requiredTargets) {
        this(relevantTargetFilter, stuckScoreThreshold, requiredTargets, 0, null);
    }

    private CASGovernor(
            Predicate<String> relevantTargetFilter,
            int stuckScoreThreshold,
            Set<String> requiredTargets,
            int maxTransientRetries,
            Set<String> transientErrorTypes
    ) {
        this.relevantTargetFilter = relevantTargetFilter == null ? (target -> true) : relevantTargetFilter;
        this.stuckScoreThreshold = stuckScoreThreshold;
        this.requiredTargets = requiredTargets;
        this.maxTransientRetries = maxTransientRetries;
        this.transientErrorTypes = (transientErrorTypes == null) ? Set.of() : transientErrorTypes;
    }

    /**
     * Returns a NEW CASGovernor with the same relevance/target configuration
     * as this one, but with bounded-retry mode enabled: a failed action
     * whose failures are all classified as one of {@code transientErrorTypes}
     * may be retried up to {@code maxTransientRetries} times before being
     * hard-blocked, rather than blocked on the first repeat. See the
     * {@link #maxTransientRetries} field doc for the full rationale.
     *
     * Composes with every existing factory, e.g.:
     * {@code CASGovernor.forAnyOfExpectedTargets(targets).withBoundedRetry(2, Set.of("TRANSIENT_READ_ERROR"))}
     */
    public CASGovernor withBoundedRetry(int maxTransientRetries, Set<String> transientErrorTypes) {
        return new CASGovernor(
                this.relevantTargetFilter,
                this.stuckScoreThreshold,
                this.requiredTargets,
                maxTransientRetries,
                transientErrorTypes
        );
    }

    /** Convenience factory for harnesses that know the single correct file target. */
    public static CASGovernor forExpectedTarget(String expectedTarget) {
        String normalizedExpected = normalizeTargetStatic(expectedTarget);
        return new CASGovernor(target -> normalizeTargetStatic(target).equals(normalizedExpected));
    }

    /** Convenience factory combining a known target with a custom STUCK threshold. */
    public static CASGovernor forExpectedTarget(String expectedTarget, int stuckScoreThreshold) {
        String normalizedExpected = normalizeTargetStatic(expectedTarget);
        return new CASGovernor(
                target -> normalizeTargetStatic(target).equals(normalizedExpected),
                stuckScoreThreshold
        );
    }

    /**
     * Factory for tasks whose answer requires reading MULTIPLE specific
     * files (e.g. two files that must be synthesized together). RELEASED
     * only fires once every target in {@code requiredTargets} has been
     * successfully read at least once -- see the {@link #requiredTargets}
     * field doc for the bug this exists to fix.
     */
    public static CASGovernor forExpectedTargets(java.util.Collection<String> requiredTargets) {
        Set<String> normalized = new java.util.LinkedHashSet<>();
        for (String target : requiredTargets) {
            normalized.add(normalizeTargetStatic(target));
        }
        return new CASGovernor(target -> true, DEFAULT_STUCK_SCORE_THRESHOLD, normalized);
    }

    /**
     * Factory for tasks where MULTIPLE files each independently suffice --
     * e.g. a primary source and a reliable backup covering the same
     * information (see MockTaskSet's H005). RELEASED fires as soon as ANY
     * ONE of {@code acceptableTargets} has been successfully read -- OR
     * semantics, the mirror image of {@link #forExpectedTargets} (AND
     * semantics, for tasks needing every listed file). Reuses the existing
     * single-predicate relevantTargetFilter mechanism unchanged (this is
     * just a different predicate, not a new code path), so it carries zero
     * risk to any other constructor/factory.
     */
    public static CASGovernor forAnyOfExpectedTargets(java.util.Collection<String> acceptableTargets) {
        Set<String> normalized = new java.util.LinkedHashSet<>();
        for (String target : acceptableTargets) {
            normalized.add(normalizeTargetStatic(target));
        }
        return new CASGovernor(target -> normalized.contains(normalizeTargetStatic(target)));
    }

    public RegulatorySignal evaluate(List<AgentEvent> history) {
        BurdenBreakdown breakdown = computeBurdenBreakdown(history);
        int burden = breakdown.burden();

        if (hasRecoveredUsefulContent(history)) {
            return new RegulatorySignal(
                    RegulatoryState.RELEASED,
                    burden,
                    "Recovery succeeded. Do not use more tools unless absolutely necessary. Produce FINAL_ANSWER."
            );
        }

        if (breakdown.stuckScore() >= stuckScoreThreshold) {
            return new RegulatorySignal(
                    RegulatoryState.STUCK,
                    burden,
                    "Repeated failure or looping is occurring. Stop the current strategy entirely and choose a "
                            + "genuinely different uncertainty-reducing action; retrying variations of what has "
                            + "already failed will not help."
            );
        }

        if (hasRepeatedFailedAction(history)) {
            return new RegulatorySignal(
                    RegulatoryState.RECOVERING,
                    burden,
                    "Repeated failure detected. Stop repeating failed assumptions. Prefer inspection, schema/listing, or clarification."
            );
        }

        if (hasRepeatedSuccessfulAction(history)) {
            return new RegulatorySignal(
                    RegulatoryState.RECOVERING,
                    burden,
                    "Repeated successful action detected. Use information already in history instead of repeating the same tool call. "
                  + "If a directory is empty or a path is invalid, inspect the workspace root using target."
            );
        }

        if (hasSuccessfulUncertaintyReductionAfterFailure(history)) {
            return new RegulatorySignal(
                    RegulatoryState.RECOVERING,
                    burden,
                    "Recovery is underway. A successful uncertainty-reducing action occurred after disturbance. Continue using the recovered information toward task completion."
            );
        }

        if (hasAnyFailure(history)) {
            return new RegulatorySignal(
                    RegulatoryState.DISTURBED,
                    burden,
                    "A disturbance occurred. Avoid blind retries. Use the next action to reduce uncertainty."
            );
        }

        return new RegulatorySignal(
                RegulatoryState.NORMAL,
                burden,
                "Normal operation. Proceed with the task."
        );
    }
    
    private boolean hasSuccessfulUncertaintyReductionAfterFailure(List<AgentEvent> history) {
        boolean seenFailure = false;

        for (AgentEvent event : history) {
            if (event.isFailedToolCall()) {
                seenFailure = true;
                continue;
            }

            if (!seenFailure) {
                continue;
            }

            if (isSuccessfulUncertaintyReducingAction(event)) {
                return true;
            }
        }

        return false;
    }

    private boolean isSuccessfulUncertaintyReducingAction(AgentEvent event) {
        if (!event.isSuccessfulToolCall()) {
            return false;
        }

        ToolAction action = event.action();

        if (action == null) {
            return false;
        }

        if (!"file".equalsIgnoreCase(action.tool())) {
            return false;
        }

        return "LIST_DIRECTORY".equalsIgnoreCase(action.actionType());
    }
    public GovernorDecision reviewAction(List<AgentEvent> history, ToolAction proposed) {
        return reviewAction(evaluate(history), history, proposed);
    }

    /**
     * Same review logic as {@link #reviewAction(List, ToolAction)}, but takes the
     * regulatory signal as an explicit parameter instead of recomputing it from
     * {@code history}. This exists so a caller can drive the state-dependent parts
     * of blocking (RELEASED / recovery-mode warning) from a signal that did NOT
     * come from evaluating this exact history -- e.g. a lagged signal (DF timing)
     * or a shuffled signal from an unrelated episode (the sham/placebo condition,
     * see {@code RunStochasticPolicyExperiment}'s {@code sham_regulation}
     * condition). The literal-repetition checks below still correctly use the
     * real {@code history}, since those aren't part of what a sham test is
     * checking -- only whether STATE-CONTINGENT signal timing matters.
     */
    public GovernorDecision reviewAction(RegulatorySignal signal, List<AgentEvent> history, ToolAction proposed) {
        if (proposed == null) {
            return GovernorDecision.block(
                    "Blocked null action.",
                    signal
            );
        }

        if (signal.isReleased() && !"none".equalsIgnoreCase(proposed.tool())) {
            return GovernorDecision.block(
                    "CAS release: useful content has already been recovered. "
                            + "Do not continue tool use; produce FINAL_ANSWER.",
                    signal
            );
        }

        if (isBlockedRepeat(history, proposed)) {
            return GovernorDecision.block(
                    "CAS recovery modulation: proposed action repeats a failed action. "
                            + "Do not retry the same failed assumption. Choose an uncertainty-reducing action instead.",
                    signal
            );
        }

        if (wouldRepeatSuccessfulAction(history, proposed)) {
            return GovernorDecision.block(
                    "CAS burden modulation: proposed action repeats a successful action. "
                            + "Use the previous result already in history.",
                    signal
            );
        }

        if (signal.isRecoveryMode()) {
            return GovernorDecision.warn(
                    "CAS recovery mode active: action allowed, but avoid blind retry and reduce uncertainty.",
                    signal
            );
        }

        return GovernorDecision.allow("Action allowed.", signal);
    }

    public GovernorDecision reviewFinalAnswer(List<AgentEvent> history, String finalAnswer) {
        return reviewFinalAnswer(evaluate(history), history, finalAnswer);
    }

    /** See {@link #reviewAction(RegulatorySignal, List, ToolAction)} for why this overload exists. */
    public GovernorDecision reviewFinalAnswer(RegulatorySignal signal, List<AgentEvent> history, String finalAnswer) {
        if (finalAnswer == null || finalAnswer.isBlank()) {
            return GovernorDecision.block(
                    "Blocked empty final answer.",
                    signal
            );
        }

        if (!hasRecoveredUsefulContent(history) && hasAnyFailure(history)) {
            return GovernorDecision.block(
                    "CAS blocked premature final answer: there were tool failures, "
                            + "but no successful recovery content has been obtained.",
                    signal
            );
        }

        return GovernorDecision.allow("Final answer allowed.", signal);
    }

    public void observe(ToolAction action, ToolResult result) {
        // Placeholder for later adaptive state memory.
        // Current implementation is history-based and deterministic.
    }

    /**
     * @param failed             total failed tool calls.
     * @param repeatedFailed     failed tool calls whose (tool, actionType, target) key
     *                           was already seen in an earlier failure.
     * @param repeatedSuccessful successful tool calls whose key was already seen in an
     *                           earlier success.
     * @param postErrorToolCalls tool calls made after the first failure in the episode,
     *                           regardless of whether they repeat anything.
     */
    private record BurdenBreakdown(
            int failed,
            int repeatedFailed,
            int repeatedSuccessful,
            int postErrorToolCalls
    ) {
        /** Reported burden: includes raw elapsed post-error activity, for observability/metrics. */
        int burden() {
            return failed
                    + 3 * repeatedFailed
                    + 2 * repeatedSuccessful
                    + Math.max(0, postErrorToolCalls - 1);
        }

        /**
         * Repetition-only score used to decide STUCK: deliberately excludes the
         * elapsed-post-error-steps term so that a long but non-repeating recovery
         * (e.g. exploring several distinct directories) isn't misclassified as
         * looping just because it took many steps.
         */
        int stuckScore() {
            return 3 * repeatedFailed + 2 * repeatedSuccessful;
        }
    }

    private BurdenBreakdown computeBurdenBreakdown(List<AgentEvent> history) {
        int failed = 0;
        int repeatedFailed = 0;
        int repeatedSuccessful = 0;
        int postErrorToolCalls = 0;
        boolean afterFirstFailure = false;

        Set<String> failedKeys = new HashSet<>();
        Set<String> successfulKeys = new HashSet<>();

        for (AgentEvent event : history) {
            if (!event.isToolCall()) {
                continue;
            }

            if (afterFirstFailure) {
                postErrorToolCalls++;
            }

            String key = actionKey(event.action());

            if (event.isFailedToolCall()) {
                failed++;

                if (!afterFirstFailure) {
                    afterFirstFailure = true;
                }

                if (failedKeys.contains(key)) {
                    repeatedFailed++;
                } else {
                    failedKeys.add(key);
                }
            }

            if (event.isSuccessfulToolCall()) {
                if (successfulKeys.contains(key)) {
                    repeatedSuccessful++;
                } else {
                    successfulKeys.add(key);
                }
            }
        }

        return new BurdenBreakdown(failed, repeatedFailed, repeatedSuccessful, postErrorToolCalls);
    }

    private boolean hasAnyFailure(List<AgentEvent> history) {
        for (AgentEvent event : history) {
            if (event.isFailedToolCall()) {
                return true;
            }
        }
        return false;
    }

    private boolean hasRecoveredUsefulContent(List<AgentEvent> history) {
        if (requiredTargets != null) {
            return hasReadAllRequiredTargets(history);
        }

        for (AgentEvent event : history) {
            if (!event.isSuccessfulToolCall()) {
                continue;
            }

            ToolAction action = event.action();

            if (action == null) {
                continue;
            }

            boolean isReadFile =
                    "file".equalsIgnoreCase(action.tool())
                            && "READ_FILE".equalsIgnoreCase(action.actionType());

            boolean hasOutput =
                    event.result() != null
                            && event.result().output() != null
                            && !event.result().output().isBlank();

            boolean isRelevant = relevantTargetFilter.test(action.target());

            if (isReadFile && hasOutput && isRelevant) {
                return true;
            }
        }

        return false;
    }

    /** Multi-target mode: true only once EVERY entry in requiredTargets has been successfully read. */
    private boolean hasReadAllRequiredTargets(List<AgentEvent> history) {
        Set<String> readTargets = new HashSet<>();

        for (AgentEvent event : history) {
            if (!event.isSuccessfulToolCall()) {
                continue;
            }

            ToolAction action = event.action();

            if (action == null) {
                continue;
            }

            boolean isReadFile =
                    "file".equalsIgnoreCase(action.tool())
                            && "READ_FILE".equalsIgnoreCase(action.actionType());

            boolean hasOutput =
                    event.result() != null
                            && event.result().output() != null
                            && !event.result().output().isBlank();

            if (isReadFile && hasOutput) {
                readTargets.add(normalizeTargetStatic(action.target()));
            }
        }

        return readTargets.containsAll(requiredTargets);
    }

    private boolean hasRepeatedFailedAction(List<AgentEvent> history) {
        Set<String> seen = new HashSet<>();

        for (AgentEvent event : history) {
            if (!event.isFailedToolCall()) {
                continue;
            }

            String key = actionKey(event.action());

            if (seen.contains(key)) {
                return true;
            }

            seen.add(key);
        }

        return false;
    }

    private boolean hasRepeatedSuccessfulAction(List<AgentEvent> history) {
        Set<String> seen = new HashSet<>();

        for (AgentEvent event : history) {
            if (!event.isSuccessfulToolCall()) {
                continue;
            }

            String key = actionKey(event.action());

            if (seen.contains(key)) {
                return true;
            }

            seen.add(key);
        }

        return false;
    }

    /**
     * Entry point used by reviewAction: when bounded-retry mode is disabled
     * (maxTransientRetries &lt;= 0, true for every governor except ones built
     * via withBoundedRetry), this is IDENTICAL to the original
     * wouldRepeatFailedAction -- block any repeat of a failed action,
     * unconditionally. When enabled, delegates to
     * {@link #isBlockedRepeatWithBoundedRetry} instead.
     */
    private boolean isBlockedRepeat(List<AgentEvent> history, ToolAction proposed) {
        if (maxTransientRetries <= 0) {
            return wouldRepeatFailedAction(history, proposed);
        }
        return isBlockedRepeatWithBoundedRetry(history, proposed);
    }

    /**
     * Counts this proposed action's prior failures and classifies each by
     * errorType. If ANY prior failure for this exact action was NOT
     * classified as transient, blocks immediately -- same as the original
     * behavior, and correct for deterministic wrong-path failures (file not
     * found, wrong extension, etc.), which is most of what this codebase's
     * tasks actually test. If EVERY prior failure was transient-classified,
     * allows retrying until maxTransientRetries is reached, then blocks --
     * bounded, so a misclassified or genuinely-broken "transient" failure
     * still can't cause an infinite retry loop.
     */
    private boolean isBlockedRepeatWithBoundedRetry(List<AgentEvent> history, ToolAction proposed) {
        String proposedKey = actionKey(proposed);
        int failureCount = 0;
        boolean allTransient = true;

        for (AgentEvent event : history) {
            if (!event.isFailedToolCall()) {
                continue;
            }

            if (!proposedKey.equals(actionKey(event.action()))) {
                continue;
            }

            failureCount++;

            String errorType = (event.result() == null) ? null : event.result().errorType();
            boolean isTransient = errorType != null && transientErrorTypes.contains(errorType);

            if (!isTransient) {
                allTransient = false;
            }
        }

        if (failureCount == 0) {
            return false;
        }

        if (!allTransient) {
            return true;
        }

        return failureCount >= maxTransientRetries;
    }

    private boolean wouldRepeatFailedAction(List<AgentEvent> history, ToolAction proposed) {
        String proposedKey = actionKey(proposed);

        for (AgentEvent event : history) {
            if (!event.isFailedToolCall()) {
                continue;
            }

            if (proposedKey.equals(actionKey(event.action()))) {
                return true;
            }
        }

        return false;
    }

    private boolean wouldRepeatSuccessfulAction(List<AgentEvent> history, ToolAction proposed) {
        String proposedKey = actionKey(proposed);

        for (AgentEvent event : history) {
            if (!event.isSuccessfulToolCall()) {
                continue;
            }

            if (proposedKey.equals(actionKey(event.action()))) {
                return true;
            }
        }

        return false;
    }

    private String actionKey(ToolAction action) {
        if (action == null) {
            return "NULL_ACTION";
        }

        return normalize(action.tool())
                + "::"
                + normalize(action.actionType())
                + "::"
                + normalizeTarget(action.target());
    }

    private String normalize(String text) {
        return text == null ? "" : text.trim().toLowerCase();
    }

    private String normalizeTarget(String target) {
        return normalizeTargetStatic(target);
    }

    private static String normalizeTargetStatic(String target) {
        if (target == null || target.isBlank()) {
            return ".";
        }

        String t = target.trim();

        while (t.endsWith("/") || t.endsWith("\\")) {
            t = t.substring(0, t.length() - 1);
        }

        if (t.isBlank() || "./".equals(t)) {
            return ".";
        }

        return t;
    }
}
