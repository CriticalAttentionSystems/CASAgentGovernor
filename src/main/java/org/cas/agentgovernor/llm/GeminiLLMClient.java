/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package org.cas.agentgovernor.llm;

import com.google.genai.Client;
import com.google.genai.types.HttpOptions;
import com.google.genai.types.HttpRetryOptions;
import org.cas.agentgovernor.core.AgentEvent;
import org.cas.agentgovernor.core.ToolAction;

import java.util.List;
import java.util.Random;
/**
 *
 * @author veronique
 */


public class GeminiLLMClient implements LLMClient, AutoCloseable {

    private static final int DEFAULT_MAX_ATTEMPTS = 6;
    private static final long DEFAULT_BASE_DELAY_MILLIS = 15_000L;
    private static final long DEFAULT_MAX_DELAY_MILLIS = 120_000L;

    private final Client client;
    private final String model;
    private final int maxAttempts;
    private final long baseDelayMillis;
    private final long maxDelayMillis;
    private final Random random = new Random();
    private final AttemptGuard attemptGuard;
    private final TextGenerator generator;

    @FunctionalInterface
    public interface AttemptGuard {
        void beforeAttempt() throws Exception;
    }

    /** Package-private injection point for offline retry/budget tests. */
    @FunctionalInterface
    interface TextGenerator {
        String generate(String model, String prompt) throws Exception;
    }

    public GeminiLLMClient(String model) {
        this(model, DEFAULT_MAX_ATTEMPTS, DEFAULT_BASE_DELAY_MILLIS, DEFAULT_MAX_DELAY_MILLIS);
    }

    public GeminiLLMClient(String model, AttemptGuard attemptGuard) {
        this(model, DEFAULT_MAX_ATTEMPTS, DEFAULT_BASE_DELAY_MILLIS, DEFAULT_MAX_DELAY_MILLIS,
                attemptGuard, null);
    }

    /**
     * @param maxAttempts     how many times to retry a per-minute (RPM/TPM) 429
     *                        before giving up. Daily (RPD) 429s are never
     *                        retried regardless of this value -- see class doc
     *                        on {@link GeminiQuotaExhaustedException}.
     * @param baseDelayMillis initial backoff delay; doubles each retry (with
     *                        jitter) up to {@code maxDelayMillis}.
     */
    public GeminiLLMClient(String model, int maxAttempts, long baseDelayMillis, long maxDelayMillis) {
        this(model, maxAttempts, baseDelayMillis, maxDelayMillis, () -> {}, null);
    }

    GeminiLLMClient(String model, int maxAttempts, long baseDelayMillis, long maxDelayMillis,
            AttemptGuard attemptGuard, TextGenerator generator) {
        if (maxAttempts < 1 || baseDelayMillis < 0 || maxDelayMillis < 0) {
            throw new IllegalArgumentException("Invalid retry settings");
        }
        this.attemptGuard = java.util.Objects.requireNonNull(attemptGuard);
        // The outer loop owns retries, so every SDK request has a budget reservation.
        this.client = generator == null ? Client.builder().httpOptions(singleAttemptOptions()).build() : null;
        this.generator = generator == null
                ? (m, p) -> client.models.generateContent(m, p, null).text() : generator;
        this.model = model;
        this.maxAttempts = maxAttempts;
        this.baseDelayMillis = baseDelayMillis;
        this.maxDelayMillis = maxDelayMillis;
    }

    static HttpOptions singleAttemptOptions() {
        return HttpOptions.builder().retryOptions(HttpRetryOptions.builder().attempts(1).build()).build();
    }

    /**
     * Closes the underlying SDK Client (and its OkHttp connection pool /
     * background threads). The com.google.genai Client is documented as
     * thread-safe and shareable, so callers running many episodes should
     * construct ONE GeminiLLMClient for the whole run and close it once at
     * the end, rather than constructing (and leaking) a new one per
     * episode -- see RunGeminiTaskComparison for the corrected pattern.
     */
    @Override
    public void close() {
        if (client != null) client.close();
    }

    @Override
    public LLMResponse generate(LLMRequest request) throws Exception {
        String prompt = buildPrompt(request);

        Exception lastTransientError = null;

        for (int attempt = 1; attempt <= maxAttempts; attempt++) {
            // Deliberately outside the transient-error handlers: local budget/I/O failure
            // must abort without making a request, including on a retry.
            attemptGuard.beforeAttempt();
            try {
                String text = generator.generate(model, prompt);
                return parseResponse(text);

            } catch (com.google.genai.errors.ClientException e) {
                String message = e.getMessage();

                if (message == null || !message.contains("429")) {
                    throw e;
                }

                if (isDailyQuotaError(message)) {
                    // Not worth retrying: RPD resets at midnight Pacific time,
                    // no amount of in-process waiting fixes this today.
                    throw new GeminiQuotaExhaustedException(
                            "Gemini daily request quota (RPD) exhausted. This resets at "
                                    + "midnight Pacific time; retrying within this process "
                                    + "will not help. Original error: " + message,
                            true,
                            e
                    );
                }

                // Per-minute (RPM) or per-minute-token (TPM) limit: transient,
                // worth an exponential backoff retry.
                lastTransientError = e;

                if (attempt == maxAttempts) {
                    break;
                }

                long delay = backoffDelayMillis(attempt);

                System.err.println(
                        "Gemini per-minute rate limit hit (attempt "
                                + attempt + "/" + maxAttempts
                                + "). Backing off " + (delay / 1000) + "s before retry."
                );

                Thread.sleep(delay);

            } catch (com.google.genai.errors.ServerException e) {
                // 5xx (503 "high demand", 500, 502, 504): the server side is
                // transiently unavailable/overloaded. Its own message says
                // "usually temporary. Please try again later." -- so retry with
                // the same exponential backoff as a per-minute rate limit,
                // rather than letting it crash a long, checkpointed run.
                lastTransientError = e;

                if (attempt == maxAttempts) {
                    break;
                }

                long delay = backoffDelayMillis(attempt);

                System.err.println(
                        "Gemini transient server error (5xx) hit (attempt "
                                + attempt + "/" + maxAttempts
                                + "). Backing off " + (delay / 1000) + "s before retry."
                );

                Thread.sleep(delay);
            }
        }

        // Retries exhausted for a transient (per-minute or 5xx) condition. Reuse
        // the quota-exhausted signal (non-daily) so callers that already handle
        // it stop cleanly and resume from the checkpoint instead of crashing.
        throw new GeminiQuotaExhaustedException(
                "Gemini transient failure (per-minute rate limit or 5xx server error) persisted "
                        + "beyond " + maxAttempts + " retry attempts with exponential backoff.",
                false,
                lastTransientError
        );
    }

    /**
     * Exponential backoff with +/-20% jitter, doubling each attempt from
     * {@code baseDelayMillis}, capped at {@code maxDelayMillis}. Jitter
     * avoids every retry in a batch of parallel/sequential runs re-hitting
     * the API at exactly the same moment.
     */
    private long backoffDelayMillis(int attempt) {
        long exponential = baseDelayMillis * (1L << (attempt - 1));
        long capped = Math.min(exponential, maxDelayMillis);

        double jitterFactor = 0.8 + (0.4 * random.nextDouble()); // random value from 0.8 up to but excluding 1.2
        return (long) (capped * jitterFactor);
    }

    /**
     * Heuristic classification of a Gemini 429 error message as a daily
     * (RPD) quota exhaustion vs. a per-minute (RPM/TPM) limit. Gemini's
     * error bodies typically name the quota metric that was exceeded (e.g.
     * containing "PerDay" for daily quotas vs "PerMinute" for per-minute
     * ones); this checks for those markers case-insensitively. If neither
     * marker is present, defaults to treating it as per-minute (the more
     * common and more safely retryable case) rather than daily, so an
     * ambiguous message doesn't cause an unnecessary early abort.
     */
    private boolean isDailyQuotaError(String message) {
        String lower = message.toLowerCase();
        return lower.contains("perday")
                || lower.contains("per day")
                || lower.contains("requests per day")
                || lower.contains("rpd")
                || lower.contains("daily");
    }

    private String buildPrompt(LLMRequest request) {
        StringBuilder sb = new StringBuilder();

        sb.append("""
            You are a tool-using AI agent operating with an optional CAS regulatory signal.

            The CAS signal is not a task solution.
            It does not tell you which file, directory, value, or answer is correct.
            It only describes the current regulatory state of the interaction.

            You must answer only from tool outputs shown in the history.
            Do not invent fields, times, parameters, metrics, paths, filenames, or file contents.
            If the needed information is not visible in the tool output, propose another tool call instead of guessing.
            When producing FINAL_ANSWER, base it only on successfully read tool output.

            Tool targets must be valid relative paths inside the workspace.
            Do not use absolute paths.
            Do not use system paths.
            Do not use "/" as a target.

            You do not execute tools yourself.
            You propose exactly one next action as JSON.

            Available tools:

            1. file / READ_FILE
               target = relative file path

            2. file / LIST_DIRECTORY
               target = relative directory path

            3. none / FINAL_ANSWER
               final_answer = final response to user

            Return JSON only. No markdown. No explanation outside JSON.

            JSON schema:
            {
              "tool": "file or none",
              "action_type": "READ_FILE or LIST_DIRECTORY or FINAL_ANSWER",
              "target": "",
              "payload": "",
              "wants_final_answer": false,
              "final_answer": ""
            }

            Generic recovery rules:
            - Do not retry the same failed action unless new information in the history justifies it.
            - If an action fails, choose a low-cost action that reduces uncertainty about the available environment.
            - If a directory listing or file read provides useful task-relevant evidence, use that evidence rather than returning to the failed assumption.
            - If the needed answer is already supported by a successful file read, produce FINAL_ANSWER.

            CAS regulatory state rules:
            - NORMAL: proceed with the task normally.
            - DISTURBED: a recent action failed. Avoid blind repetition. Prefer an uncertainty-reducing action.
            - RECOVERING: useful information has been obtained after a disturbance. Continue from the recovered evidence and avoid unnecessary repeated tool calls.
            - STUCK: repeated failure or looping is occurring. Stop the current strategy and choose a different uncertainty-reducing action.
            - RELEASED: enough task-relevant evidence has been recovered. Produce FINAL_ANSWER from tool outputs only.
            """);

        sb.append("\nUser task:\n");
        sb.append(request.userTask()).append("\n\n");

        sb.append("CAS regulatory signal:\n");
        sb.append(request.regulatorySignal()).append("\n\n");

        sb.append("History:\n");
        if (request.history().isEmpty()) {
            sb.append("No prior actions.\n");
        } else {
            sb.append(this.formatHistoryForPrompt(request.history()));
            //for (AgentEvent event : request.history()) {
            //    sb.append(event.toString()).append("\n");
            //}
        }

        sb.append("\nReturn next JSON action only:\n");

        return sb.toString();
    }
    
    private String formatHistoryForPrompt(List<AgentEvent> history) {
        StringBuilder sb = new StringBuilder();

        for (AgentEvent event : history) {
            if (event.isRegulatorySignal()) {
                sb.append("REGULATORY_SIGNAL: ")
                        .append(event.regulatorySignal())
                        .append("\n");
                continue;
            }

            if (event.isToolCall()) {
                sb.append("TOOL_CALL\n");
                sb.append("tool: ").append(event.action().tool()).append("\n");
                sb.append("actionType: ").append(event.action().actionType()).append("\n");
                sb.append("target: ").append(event.action().target()).append("\n");
                sb.append("success: ").append(event.result().success()).append("\n");
                sb.append("errorType: ").append(event.result().errorType()).append("\n");
                sb.append("message: ").append(event.result().message()).append("\n");
                sb.append("output:\n");
                sb.append(event.result().output()).append("\n");
                sb.append("END_TOOL_CALL\n\n");
                continue;
            }

            if (event.eventType() == AgentEvent.EventType.GOVERNOR_INTERVENTION) {
                sb.append("GOVERNOR_EVENT: ")
                        .append(event.message())
                        .append("\n");
                continue;
            }

            if (event.eventType() == AgentEvent.EventType.FINAL_ANSWER) {
                sb.append("FINAL_ANSWER: ")
                        .append(event.message())
                        .append("\n");
            }
        }

        return sb.toString();
    }

    private LLMResponse parseResponse(String raw) {
        String cleaned = raw.trim();

        cleaned = cleaned.replace("```json", "")
                         .replace("```", "")
                         .trim();

        String tool = extractJsonString(cleaned, "tool");
        String actionType = extractJsonString(cleaned, "action_type");
        String target = extractJsonString(cleaned, "target");
        String payload = extractJsonString(cleaned, "payload");
        String finalAnswer = extractJsonString(cleaned, "final_answer");
        boolean wantsFinal = extractJsonBoolean(cleaned, "wants_final_answer");

        if ("FINAL_ANSWER".equalsIgnoreCase(actionType) || wantsFinal) {
            return new LLMResponse(
                    null,
                    true,
                    finalAnswer,
                    raw
            );
        }

        if (tool.isBlank() || actionType.isBlank()) {
            return new LLMResponse(
                    null,
                    true,
                    "Stopped: Gemini did not return a valid tool action.",
                    raw
            );
        }

        ToolAction action = new ToolAction(
                tool,
                actionType,
                target,
                payload
        );

        return new LLMResponse(
                action,
                false,
                "",
                raw
        );
    }

    /**
     * Extracts a JSON string value for {@code key} from a raw (hand-parsed,
     * not via a real JSON library) response body. Correctly skips escaped
     * characters (\", \\, \n, etc.) inside the string value -- an earlier
     * version of this method found the string's closing quote via a naive
     * "second literal double-quote character" scan, which broke as soon as
     * the model's answer contained an embedded escaped quote (e.g. a model
     * quoting a value it read, such as final_answer containing
     * {@code ...is \"unknown\".}). That bug silently truncated the answer
     * at the first escaped quote, which is exactly the pattern observed in
     * real Gemini output (answers ending in a bare trailing backslash).
     * This scans character-by-character tracking escape state instead, so
     * only a genuine, unescaped closing quote ends the string.
     */
    private String extractJsonString(String json, String key) {
        String pattern = "\"" + key + "\"";
        int keyIndex = json.indexOf(pattern);
        if (keyIndex < 0) return "";

        int colon = json.indexOf(":", keyIndex);
        if (colon < 0) return "";

        int firstQuote = json.indexOf("\"", colon + 1);
        if (firstQuote < 0) return "";

        StringBuilder value = new StringBuilder();
        boolean escaped = false;

        for (int i = firstQuote + 1; i < json.length(); i++) {
            char c = json.charAt(i);

            if (escaped) {
                switch (c) {
                    case 'n' -> value.append('\n');
                    case 't' -> value.append('\t');
                    case 'r' -> value.append('\r');
                    case '"' -> value.append('"');
                    case '\\' -> value.append('\\');
                    case '/' -> value.append('/');
                    default -> value.append(c);
                }
                escaped = false;
                continue;
            }

            if (c == '\\') {
                escaped = true;
                continue;
            }

            if (c == '"') {
                return value.toString();
            }

            value.append(c);
        }

        // Ran off the end without a closing quote (malformed/truncated
        // response) -- return what was accumulated rather than nothing,
        // consistent with the old method's behavior of degrading gracefully.
        return value.toString();
    }

    private boolean extractJsonBoolean(String json, String key) {
        String pattern = "\"" + key + "\"";
        int keyIndex = json.indexOf(pattern);
        if (keyIndex < 0) return false;

        int colon = json.indexOf(":", keyIndex);
        if (colon < 0) return false;

        String tail = json.substring(colon + 1).trim();
        return tail.startsWith("true");
    }
}
