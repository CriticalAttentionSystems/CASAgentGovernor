package org.cas.agentgovernor.experiment;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.List;
import java.util.Set;
import org.cas.agentgovernor.experiment.tasks.MockTaskSpec;
import org.cas.agentgovernor.llm.ApiBudgetExceededException;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import static org.junit.jupiter.api.Assertions.*;

class InducedRunStateTest {
    @TempDir Path root;
    static final String KEY = "T,blind,0.00,0";
    static final String ROW = KEY + ",true,true,1,0,0,0,0.50000";

    InducedRunState open(Path dir, String config, Long requested, long defaults) throws IOException {
        return new InducedRunState(dir, config, Set.of(KEY), requested, defaults);
    }

    @Test void resumesCountAndCompletedEpisodesWithoutResettingCeiling() throws Exception {
        try (var run = open(root, "config", null, 2)) {
            run.reserveAttempt();
            run.append(ROW);
        }
        try (var run = open(root, "config", null, 9000)) {
            assertTrue(run.done(KEY));
            assertEquals(1, run.attempts());
            assertEquals(2, run.limit());
            run.reserveAttempt();
            assertThrows(ApiBudgetExceededException.class, run::reserveAttempt);
            assertEquals(2, run.attempts());
        }
        try (var run = open(root, "config", 3L, 2)) {
            run.reserveAttempt();
            assertEquals(3, run.attempts());
            assertThrows(ApiBudgetExceededException.class, run::reserveAttempt);
        }
        assertThrows(IOException.class, () -> open(root, "config", 2L, 2));
    }

    @Test void interruptedEpisodeConsumesBudgetWithoutBeingMarkedDone() throws Exception {
        try (var run = open(root, "config", null, 1)) { run.reserveAttempt(); }
        try (var run = open(root, "config", null, 1)) {
            assertFalse(run.done(KEY));
            assertThrows(ApiBudgetExceededException.class, run::reserveAttempt);
        }
    }

    @Test void mockAndGeminiNamespacesAreSeparate() throws Exception {
        Path mock = InducedRunState.directory(root, "mock", "same-name");
        Path live = InducedRunState.directory(root, "gemini", "same-name");
        try (var run = open(mock, "mock", null, 1)) { run.append(ROW); }
        try (var run = open(live, "gemini", null, 1)) {
            assertFalse(run.done(KEY));
            assertEquals(0, run.attempts());
        }
    }

    @Test void changedConfigAndMissingMetadataFailClosed() throws Exception {
        try (var run = open(root, "original", null, 1)) { run.append(ROW); }
        assertThrows(IOException.class, () -> open(root, "changed", null, 1));
        // Simulate loss of a ledger, not a legitimate new run.
        Files.delete(root.resolve("run.properties"));
        assertThrows(IOException.class, () -> open(root, "original", null, 1));
        assertTrue(Files.readString(root.resolve("checkpoint.csv")).contains(ROW));
    }

    @Test void concurrentWriterIsRejectedAndLockIsReleased() throws Exception {
        try (var first = open(root, "config", null, 1)) {
            assertThrows(IOException.class, () -> open(root, "config", null, 1));
        }
        try (var second = open(root, "config", null, 1)) { assertEquals(0, second.attempts()); }
    }

    @Test void rejectsMalformedDuplicateAndOutOfGridRows() throws Exception {
        try (var run = open(root, "config", null, 1)) {
            assertThrows(IOException.class, () -> run.append(KEY + ",true,true"));
            assertThrows(IOException.class, () -> run.append(ROW.replace("0.50000", "NaN")));
            assertThrows(IOException.class, () -> run.append(ROW.replace("T,", "OTHER,")));
            assertThrows(IOException.class, () -> run.append(ROW.replace("true,true", "true,false")));
            run.append(ROW);
            assertThrows(IOException.class, () -> run.append(ROW));
        }
        Files.writeString(root.resolve("checkpoint.csv"), ROW + "\n", StandardOpenOption.APPEND);
        assertThrows(IOException.class, () -> open(root, "config", null, 1));
    }

    @Test void partialRowOnDiskIsNotSilentlySkipped() throws Exception {
        try (var run = open(root, "config", null, 1)) { run.reserveAttempt(); }
        Files.writeString(root.resolve("checkpoint.csv"), KEY, StandardOpenOption.APPEND);
        assertThrows(IOException.class, () -> open(root, "config", null, 1));
    }

    @Test void missingBudgetFieldIsNotTreatedAsZero() throws Exception {
        try (var run = open(root, "config", null, 1)) { run.reserveAttempt(); }
        Path manifest = root.resolve("run.properties");
        Files.writeString(manifest, Files.readString(manifest).replace("api_attempts=1", "api_attempts=oops"));
        assertThrows(IOException.class, () -> open(root, "config", null, 1));
    }

    @Test void rejectsUnsafeRunNamesAndUnknownCliOptions() {
        assertThrows(IllegalArgumentException.class, () -> InducedRunState.directory(root, "gemini", "../old"));
        assertThrows(IllegalArgumentException.class, () -> RunGeminiInducedCost.RunOptions.parse(new String[]{"moc"}));
        assertThrows(IllegalArgumentException.class, () -> RunGeminiInducedCost.RunOptions.parse(new String[]{"--max-api-calls", "0"}));
        assertThrows(IllegalArgumentException.class, () -> RunGeminiInducedCost.RunOptions.parse(new String[]{"--run-id"}));
        var opts = RunGeminiInducedCost.RunOptions.parse(new String[]{"mock", "--run-id", "smoke", "--max-api-calls", "12"});
        assertTrue(opts.mock());
        assertTrue(opts.explicitRunId());
        assertEquals(12L, opts.maxApiCalls());
    }

    @Test void fingerprintTracksBackendAndTaskContent() throws Exception {
        var task = new MockTaskSpec("T", "prompt", "file", "answer", java.util.Map.of("file", "content"));
        var changed = new MockTaskSpec("T", "new prompt", "file", "answer", java.util.Map.of("file", "content"));
        String original = RunGeminiInducedCost.configuration(false, List.of(task));
        assertEquals(original, RunGeminiInducedCost.configuration(false, List.of(task)));
        assertNotEquals(InducedRunState.fingerprint(original), InducedRunState.fingerprint(
                RunGeminiInducedCost.configuration(true, List.of(task))));
        assertNotEquals(original, RunGeminiInducedCost.configuration(false, List.of(changed)));
    }
}
