/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package org.cas.agentgovernor.llm;

/**
 * Thrown when the Gemini API refuses a request due to quota exhaustion that
 * retrying-with-backoff cannot fix within this process's lifetime.
 *
 * Gemini 429 responses come in (at least) two flavors that require very
 * different handling:
 *
 *   - Per-minute (RPM) or per-minute-token (TPM) limits: transient, worth a
 *     short exponential backoff and retry within the same run.
 *   - Per-day (RPD) limits: not worth retrying at all until the quota resets
 *     at midnight Pacific time. A caller that keeps retrying this either
 *     busy-loops uselessly or (worse) sleeps for a long time and then fails
 *     anyway, burning wall-clock time without making progress.
 *
 * {@link #dailyQuotaExhausted} lets a harness (e.g. RunGeminiTaskComparison)
 * distinguish the two: on a daily-quota exhaustion, the right move is to
 * checkpoint whatever has been completed so far and stop the whole run
 * immediately, with a clear message about when to resume -- not to keep
 * retrying or to lose the completed work by crashing uncleanly.
 *
 * @author veronique
 */
public class GeminiQuotaExhaustedException extends Exception {

    private final boolean dailyQuotaExhausted;

    public GeminiQuotaExhaustedException(String message, boolean dailyQuotaExhausted) {
        super(message);
        this.dailyQuotaExhausted = dailyQuotaExhausted;
    }

    public GeminiQuotaExhaustedException(String message, boolean dailyQuotaExhausted, Throwable cause) {
        super(message, cause);
        this.dailyQuotaExhausted = dailyQuotaExhausted;
    }

    /**
     * @return true if this was a per-day (RPD) quota exhaustion (resets at
     *         midnight Pacific time, not worth retrying today); false if
     *         this was a per-minute (RPM/TPM) limit that was retried with
     *         backoff until {@code maxAttempts} was exhausted.
     */
    public boolean isDailyQuotaExhausted() {
        return dailyQuotaExhausted;
    }
}
