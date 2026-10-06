package org.cas.agentgovernor.llm;

/** A local, persistent request-attempt budget is exhausted. Not a provider quota. */
public final class ApiBudgetExceededException extends Exception {
    public ApiBudgetExceededException(long used, long limit) {
        super("API attempt budget exhausted (" + used + "/" + limit
                + "). Resuming does not reset it; explicitly increase --max-api-calls if intended.");
    }
}
