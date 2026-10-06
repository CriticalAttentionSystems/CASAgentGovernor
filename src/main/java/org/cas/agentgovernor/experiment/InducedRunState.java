package org.cas.agentgovernor.experiment;

import java.io.IOException;
import java.io.StringReader;
import java.io.StringWriter;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.channels.FileLock;
import java.nio.channels.OverlappingFileLockException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HashSet;
import java.util.HexFormat;
import java.util.Properties;
import java.util.Set;
import org.cas.agentgovernor.llm.ApiBudgetExceededException;

/** Exclusive run ownership, validated checkpoint, and durable pre-request reservations. */
final class InducedRunState implements AutoCloseable {
    static final String HEADER = "task_id,mode,alpha,repeat,success,answered,post_error_tool_burden,"
            + "hard_blocks,induced_failures,backoff_steps,mean_regulation_intensity";
    private final Path directory;
    private final FileChannel lockChannel;
    private final FileLock lock;
    private final Properties state = new Properties();
    private final Set<String> expectedKeys;
    private final Set<String> completed = new HashSet<>();
    private long attempts;
    private long limit;

    static Path directory(Path root, String backend, String runId) {
        if (!Set.of("mock", "gemini").contains(backend)
                || !runId.matches("[A-Za-z0-9][A-Za-z0-9_-]{0,79}")) {
            throw new IllegalArgumentException("Invalid backend or run ID (use letters, digits, _ and -).");
        }
        return root.resolve("gemini_induced_runs").resolve(backend).resolve(runId);
    }

    static String fingerprint(String configuration) {
        try {
            return HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256")
                    .digest(configuration.getBytes(StandardCharsets.UTF_8)));
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException(e);
        }
    }

    InducedRunState(Path directory, String configuration, Set<String> expectedKeys,
            Long requestedLimit, long defaultLimit) throws IOException {
        this.directory = directory;
        this.expectedKeys = Set.copyOf(expectedKeys);
        Files.createDirectories(directory);
        lockChannel = FileChannel.open(directory.resolve("run.lock"),
                StandardOpenOption.CREATE, StandardOpenOption.WRITE);
        FileLock acquired;
        try {
            acquired = lockChannel.tryLock();
        } catch (OverlappingFileLockException e) {
            lockChannel.close();
            throw new IOException("Run already in use: " + directory, e);
        }
        if (acquired == null) {
            lockChannel.close();
            throw new IOException("Run already in use: " + directory);
        }
        lock = acquired;
        try {
            Path manifest = directory.resolve("run.properties");
            if (Files.exists(manifest)) {
                state.load(new StringReader(Files.readString(manifest)));
                if (!configuration.equals(state.getProperty("configuration"))
                        || !fingerprint(configuration).equals(state.getProperty("fingerprint"))) {
                    throw new IOException("Run configuration/implementation mismatch; use a new run ID.");
                }
                attempts = Long.parseLong(state.getProperty("api_attempts"));
                limit = Long.parseLong(state.getProperty("api_limit"));
                if (attempts < 0 || limit < 1 || attempts > limit) {
                    throw new IOException("Invalid persistent API budget.");
                }
                loadCheckpoint();
            } else {
                if (Files.exists(checkpoint())) {
                    throw new IOException("Checkpoint lacks provenance/budget metadata; refusing to adopt it.");
                }
                attempts = 0;
                limit = defaultLimit;
                state.setProperty("configuration", configuration);
                state.setProperty("fingerprint", fingerprint(configuration));
            }
            if (requestedLimit != null) {
                limit = requestedLimit;
            }
            if (limit < 1 || limit < attempts) {
                throw new IOException("API limit must be positive and at least the number of reserved attempts.");
            }
            if (!Files.exists(manifest)) {
                Files.writeString(checkpoint(), HEADER + "\n", StandardOpenOption.CREATE_NEW);
            }
            persist();
        } catch (Exception e) {
            close();
            throw new IOException("Cannot safely open run " + directory + ": " + e.getMessage(), e);
        }
    }

    Path checkpoint() { return directory.resolve("checkpoint.csv"); }
    Path workspaceRoot() { return directory.resolve("workspaces"); }
    synchronized long attempts() { return attempts; }
    long limit() { return limit; }
    boolean done(String key) { return completed.contains(key); }

    /** Persist before sending; a crash after reservation may overcount but cannot refund a call. */
    synchronized void reserveAttempt() throws IOException, ApiBudgetExceededException {
        if (!lock.isValid()) throw new IOException("Run is closed.");
        if (attempts >= limit) throw new ApiBudgetExceededException(attempts, limit);
        attempts++;
        persist();
    }

    void append(String row) throws IOException {
        String key = validateRow(row);
        if (completed.contains(key)) throw new IOException("Duplicate checkpoint key: " + key);
        try (FileChannel out = FileChannel.open(checkpoint(), StandardOpenOption.WRITE,
                StandardOpenOption.APPEND)) {
            ByteBuffer bytes = StandardCharsets.UTF_8.encode(row + "\n");
            while (bytes.hasRemaining()) out.write(bytes);
            out.force(true);
        }
        completed.add(key);
    }

    private void loadCheckpoint() throws IOException {
        String text = Files.readString(checkpoint());
        if (!text.endsWith("\n")) throw new IOException("Truncated checkpoint; refusing to skip a partial row.");
        String[] lines = text.split("\n");
        if (lines.length == 0 || !HEADER.equals(lines[0])) throw new IOException("Wrong checkpoint schema.");
        for (int i = 1; i < lines.length; i++) {
            String key = validateRow(lines[i]);
            if (!completed.add(key)) throw new IOException("Duplicate checkpoint key: " + key);
        }
    }

    private String validateRow(String row) throws IOException {
        try {
            String[] f = row.split(",", -1);
            if (f.length != 11) throw new IllegalArgumentException("expected 11 columns");
            String key = String.join(",", f[0], f[1], f[2], f[3]);
            if (!expectedKeys.contains(key)) throw new IllegalArgumentException("unknown episode key");
            for (int i = 4; i <= 5; i++) {
                if (!Set.of("true", "false").contains(f[i])) throw new IllegalArgumentException("invalid boolean");
            }
            if (f[4].equals("true") && f[5].equals("false")) throw new IllegalArgumentException("success without answer");
            for (int i = 6; i <= 9; i++) {
                if (Integer.parseInt(f[i]) < 0) throw new IllegalArgumentException("negative count");
            }
            double g = Double.parseDouble(f[10]);
            if (!Double.isFinite(g) || g < 0 || g > 1) throw new IllegalArgumentException("invalid intensity");
            return key;
        } catch (IllegalArgumentException e) {
            throw new IOException("Invalid checkpoint row: " + e.getMessage(), e);
        }
    }

    private void persist() throws IOException {
        state.setProperty("api_attempts", Long.toString(attempts));
        state.setProperty("api_limit", Long.toString(limit));
        StringWriter text = new StringWriter();
        state.store(text, "Do not reset counts or copy results from a different run");
        Path temp = Files.createTempFile(directory, "state-", ".tmp");
        try {
            try (FileChannel out = FileChannel.open(temp, StandardOpenOption.WRITE)) {
                ByteBuffer bytes = StandardCharsets.UTF_8.encode(text.toString());
                while (bytes.hasRemaining()) out.write(bytes);
                out.force(true);
            }
            // Fail closed if this filesystem cannot atomically replace the budget ledger.
            Files.move(temp, directory.resolve("run.properties"),
                    StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING);
        } finally {
            Files.deleteIfExists(temp);
        }
    }

    @Override public void close() throws IOException {
        try { lock.release(); } finally { lockChannel.close(); }
    }
}
