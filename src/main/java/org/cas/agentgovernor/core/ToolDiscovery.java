/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package org.cas.agentgovernor.core;

import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

/**
 * Shared, non-omniscient exploration heuristics used by every mock
 * policy/LLM in this codebase (the stochastic {@code ActionPolicy},
 * {@code MockClingingLLM}, {@code MockPersistentLLM}).
 *
 * Every method here operates ONLY on information that has actually
 * appeared in the agent's own tool-call history (i.e. what a real prior
 * successful LIST_DIRECTORY call reported) — never on a benchmark
 * harness's known-correct {@code expectedTarget} path. "Recovery" and
 * "candidate selection" therefore mean genuine discovery, the same way a
 * real agent would have to find things, not privileged path knowledge
 * smuggled in from the test harness.
 *
 * This was previously duplicated (with the omniscience bug) independently
 * in three places; it's centralized here so a fix in one place fixes all
 * three, and so new mock policies/LLMs get the correct behavior by default.
 *
 * @author veronique
 */
public final class ToolDiscovery {

    private ToolDiscovery() {
    }

    public record DiscoveredListing(
            List<String> directories,
            List<String> files,
            Set<String> listedDirectories
    ) {
    }

    /**
     * Replays the history's successful LIST_DIRECTORY calls to reconstruct
     * what the caller actually knows: which directories have already been
     * listed, and which subdirectories/files those listings reported (in
     * discovery order).
     */
    public static DiscoveredListing collectDiscoveredEntries(List<AgentEvent> history) {
        List<String> directories = new ArrayList<>();
        List<String> files = new ArrayList<>();
        Set<String> seenDirectories = new LinkedHashSet<>();
        Set<String> seenFiles = new LinkedHashSet<>();
        Set<String> listedDirectories = new LinkedHashSet<>();

        for (AgentEvent event : history) {
            if (!event.isSuccessfulToolCall()) {
                continue;
            }

            ToolAction action = event.action();

            if (action == null
                    || !"file".equalsIgnoreCase(action.tool())
                    || !"LIST_DIRECTORY".equalsIgnoreCase(action.actionType())) {
                continue;
            }

            String dir = normalizeTarget(action.target());
            listedDirectories.add(dir);

            String output = event.result() == null ? null : event.result().output();

            if (output == null || output.isBlank()) {
                continue;
            }

            for (String line : output.split("\n")) {
                String entry = line.trim();

                if (entry.isEmpty() || "(empty directory)".equals(entry)) {
                    continue;
                }

                boolean isDirectory = entry.endsWith("/");
                String name = isDirectory ? entry.substring(0, entry.length() - 1) : entry;
                String path = ".".equals(dir) ? name : dir + "/" + name;
                String normalizedPath = normalizeTarget(path);

                if (isDirectory) {
                    if (seenDirectories.add(normalizedPath)) {
                        directories.add(normalizedPath);
                    }
                } else if (seenFiles.add(normalizedPath)) {
                    files.add(normalizedPath);
                }
            }
        }

        return new DiscoveredListing(directories, files, listedDirectories);
    }

    /**
     * Picks a subdirectory to inspect next, drawn ONLY from directories
     * reported in prior successful LIST_DIRECTORY outputs (breadth-first
     * over the discovered tree). Returns "" if there is nothing left to
     * explore given what has been discovered so far.
     */
    public static String nextDiscoveredUnlistedDirectory(List<AgentEvent> history) {
        DiscoveredListing discovered = collectDiscoveredEntries(history);

        for (String candidate : discovered.directories()) {
            if (!discovered.listedDirectories().contains(candidate)) {
                return candidate;
            }
        }

        return "";
    }

    /**
     * Picks a file to read from ones actually reported in prior successful
     * LIST_DIRECTORY outputs, scored by how many meaningful tokens it
     * shares with the task prompt, preferring the best-scoring unread
     * candidate. Falls back to any discovered-but-unread file if nothing
     * scores above zero. Returns null if nothing has been discovered yet.
     *
     * Some tasks are deliberately adversarial to plain keyword overlap
     * (e.g. a decoy file whose name shares more prompt words than the
     * correct one, or two files differing only by an embedded date);
     * getting those wrong sometimes is a legitimate outcome of using only
     * discoverable information, not a bug in this heuristic.
     */
    public static String discoveredReadCandidateTarget(String userTask, List<AgentEvent> history) {
        DiscoveredListing discovered = collectDiscoveredEntries(history);
        Set<String> alreadyRead = readTargets(history);
        Set<String> promptKeywords = tokenize(userTask);
        boolean wantsLatest = userTask != null && userTask.toLowerCase().contains("latest");

        String best = null;
        int bestScore = -1;

        for (String file : discovered.files()) {
            if (alreadyRead.contains(normalizeTarget(file))) {
                continue;
            }

            int score = overlapScore(promptKeywords, tokenize(file));

            if (score > bestScore
                    || (score == bestScore && wantsLatest && best != null && file.compareTo(best) > 0)) {
                best = file;
                bestScore = score;
            }
        }

        return best;
    }

    public static Set<String> readTargets(List<AgentEvent> history) {
        Set<String> targets = new LinkedHashSet<>();

        for (AgentEvent event : history) {
            if (!event.isSuccessfulToolCall()) {
                continue;
            }

            ToolAction action = event.action();

            if (action == null
                    || !"file".equalsIgnoreCase(action.tool())
                    || !"READ_FILE".equalsIgnoreCase(action.actionType())) {
                continue;
            }

            targets.add(normalizeTarget(action.target()));
        }

        return targets;
    }

    private static final Set<String> STOPWORDS = Set.of(
            "the", "a", "an", "is", "are", "be", "been", "and", "or", "not", "no",
            "may", "have", "has", "had", "with", "from", "this", "that", "of", "to",
            "in", "on", "at", "it", "its", "if", "so", "for", "as", "by", "use",
            "read", "find", "report", "summarize", "check", "inspect", "file",
            "files", "probably", "please", "there", "multiple", "unrelated"
    );

    public static int overlapScore(Set<String> promptKeywords, Set<String> candidateTokens) {
        int score = 0;

        for (String token : candidateTokens) {
            if (promptKeywords.contains(token)) {
                score++;
            }
        }

        return score;
    }

    public static Set<String> tokenize(String text) {
        if (text == null || text.isBlank()) {
            return Set.of();
        }

        Set<String> tokens = new LinkedHashSet<>();

        for (String raw : text.toLowerCase().split("[^a-z0-9]+")) {
            if (raw.isBlank() || STOPWORDS.contains(raw)) {
                continue;
            }

            tokens.add(raw);
        }

        return tokens;
    }

    public static String normalizeTarget(String target) {
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
