package org.cas.agentgovernor.llm;

import java.io.IOException;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import com.google.genai.errors.ServerException;
import com.google.genai.errors.ClientException;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/** No SDK client, credentials, network, or sleep is used by these tests. */
class GeminiBudgetTest {
    static final LLMRequest REQUEST = new LLMRequest("test", List.of());
    static final String ANSWER = "{\"action_type\":\"FINAL_ANSWER\",\"final_answer\":\"done\"}";

    @Test void every503RetryReservesBudgetAndExhaustionPreventsNextSend() throws Exception {
        AtomicInteger reserved = new AtomicInteger();
        AtomicInteger sends = new AtomicInteger();
        try (var client = new GeminiLLMClient("fake", 6, 0, 0, () -> {
            if (reserved.get() == 2) throw new ApiBudgetExceededException(2, 2);
            reserved.incrementAndGet();
        }, (model, prompt) -> {
            assertEquals(sends.get() + 1, reserved.get());
            sends.incrementAndGet();
            throw new ServerException(503, "UNAVAILABLE", "overloaded");
        })) {
            assertThrows(ApiBudgetExceededException.class, () -> client.generate(REQUEST));
            assertEquals(2, sends.get());
            assertEquals(2, reserved.get());
        }
    }

    @Test void a429RetryAndNextDecisionBothConsumeBudget() throws Exception {
        AtomicInteger reserved = new AtomicInteger();
        AtomicInteger sends = new AtomicInteger();
        try (var client = new GeminiLLMClient("fake", 6, 0, 0, reserved::incrementAndGet,
                (model, prompt) -> {
                    if (sends.incrementAndGet() == 1) throw new ClientException(429, "RESOURCE_EXHAUSTED", "429 PerMinute");
                    return ANSWER;
                })) {
            assertEquals("done", client.generate(REQUEST).finalAnswer());
            assertEquals("done", client.generate(REQUEST).finalAnswer());
            assertEquals(3, reserved.get());
            assertEquals(3, sends.get());
        }
    }

    @Test void ledgerFailurePreventsAnySend() throws Exception {
        AtomicInteger sends = new AtomicInteger();
        try (var client = new GeminiLLMClient("fake", 6, 0, 0,
                () -> { throw new IOException("disk full"); },
                (model, prompt) -> { sends.incrementAndGet(); return ANSWER; })) {
            assertThrows(IOException.class, () -> client.generate(REQUEST));
            assertEquals(0, sends.get());
        }
    }

    @Test void zeroRemainingBudgetPreventsFirstSend() throws Exception {
        try (var client = new GeminiLLMClient("fake", 6, 0, 0,
                () -> { throw new ApiBudgetExceededException(1, 1); },
                (model, prompt) -> { fail("No request should be sent"); return ANSWER; })) {
            assertThrows(ApiBudgetExceededException.class, () -> client.generate(REQUEST));
        }
    }

    @Test void sdkDoesNotAddUncountedRetries() {
        assertEquals(1, GeminiLLMClient.singleAttemptOptions().retryOptions().orElseThrow()
                .attempts().orElseThrow());
    }
}
