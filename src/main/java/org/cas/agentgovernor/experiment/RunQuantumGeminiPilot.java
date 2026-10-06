/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package org.cas.agentgovernor.experiment;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.EnumSet;
import java.util.Set;

import org.cas.agentgovernor.agents.QuantumGovernedLLMAgent;
import org.cas.agentgovernor.core.CASGovernor;
import org.cas.agentgovernor.core.RegulatoryState;
import org.cas.agentgovernor.experiment.tasks.MockTaskSet;
import org.cas.agentgovernor.experiment.tasks.MockTaskSpec;
import org.cas.agentgovernor.llm.GeminiLLMClient;
import org.cas.agentgovernor.metrics.AgentRunMetrics;
import org.cas.agentgovernor.tools.FileTool;
import org.cas.agentgovernor.tools.FlakyFileTool;
import org.cas.agentgovernor.tools.ToolRegistry;

/**
 * Multi-seed real-Gemini pilot for {@link QuantumGovernedLLMAgent} on H005,
 * using the SAME setup {@code RunQuantumGeminiDemo} already validated
 * (adaptive/bounded-retry governor, recalibrated signal thresholds,
 * mindfulness stabilization wired in). Deliberately concise per-episode
 * output (one line each) instead of full transcripts -- the open question
 * this exists to answer is whether RECOVERING/STUCK ever actually fire on
 * real data, and what the real success rate looks like across more than
 * one episode, not to re-print full transcripts N times.
 *
 * <p>SEEDS is kept deliberately small (5): this spends real Gemini quota
 * per episode, and the point here is a first real read on repeatability,
 * not a statistically powered sweep -- same caution already applied to
 * every other real-Gemini run in this codebase.
 *
 * <p>No checkpointing: if this is interrupted partway through, already-
 * printed episodes are still visible in your terminal/log, but a re-run
 * starts over from seed 0. Add checkpointing (mirroring
 * {@code RunGeminiTaskComparison}'s pattern) if this needs to scale up
 * later.
 *
 * @author veronique
 */
public class RunQuantumGeminiPilot {

    private static final Path WORKSPACE = Path.of("test_workspace_quantum_gemini_pilot");
    private static final int MAX_STEPS = 8;
    private static final int SEEDS = 5;
    private static final long PAUSE_BETWEEN_RUNS_MILLIS = 15_000L;

    private static final int ADAPTIVE_MAX_TRANSIENT_RETRIES = 2;
    private static final Set<String> ADAPTIVE_TRANSIENT_ERROR_TYPES = Set.of("TRANSIENT_READ_ERROR");

    public static void main(String[] args) throws Exception {
        MockTaskSpec task = findH005();

        System.out.println("Quantum-governed Gemini pilot on H005 -- " + SEEDS
                + " real episodes, max " + MAX_STEPS + " steps each.");
        System.out.println("=".repeat(78));

        int successes = 0;
        long totalStepsOnSuccess = 0;
        int episodesReachingRecovering = 0;
        int episodesReachingStuck = 0;

        try (GeminiLLMClient llm = new GeminiLLMClient("gemini-2.5-flash")) {
            for (int seed = 0; seed < SEEDS; seed++) {
                resetWorkspace(task);

                ToolRegistry tools = buildTools(task);
                CASGovernor governor = buildAdaptiveGovernor(task);

                QuantumGovernedLLMAgent agent = new QuantumGovernedLLMAgent(
                        llm, governor, tools, 8_000L, /* quantumSeed = */ seed
                );

                String answer = agent.run(task.prompt(), MAX_STEPS);

                AgentRunMetrics metrics =
                        AgentRunMetrics.fromHistory("quantum_governed_gemini_h005_pilot", agent.history(), task.expectedAnswer());

                boolean success = metrics.success();
                int stepsTaken = agent.entropyHistory().size();
                double finalEntropy = agent.entropyHistory().isEmpty() ? 0.0
                        : agent.entropyHistory().get(agent.entropyHistory().size() - 1);
                double finalMu = agent.muHistory().isEmpty() ? 0.0
                        : agent.muHistory().get(agent.muHistory().size() - 1);

                Set<RegulatoryState> statesReached = EnumSet.noneOf(RegulatoryState.class);
                agent.controlSignalHistory().forEach(sig -> statesReached.add(sig.state()));

                boolean reachedRecovering = statesReached.contains(RegulatoryState.RECOVERING);
                boolean reachedStuck = statesReached.contains(RegulatoryState.STUCK);
                if (reachedRecovering) {
                    episodesReachingRecovering++;
                }
                if (reachedStuck) {
                    episodesReachingStuck++;
                }

                if (success) {
                    successes++;
                    totalStepsOnSuccess += stepsTaken;
                }

                System.out.printf(
                        "seed %d: %-7s steps=%d finalEntropy=%.4f finalMu=%.4f statesReached=%s answer=%s%n",
                        seed,
                        success ? "SUCCESS" : "FAIL",
                        stepsTaken,
                        finalEntropy,
                        finalMu,
                        statesReached,
                        truncate(answer, 60)
                );

                if (seed < SEEDS - 1) {
                    Thread.sleep(PAUSE_BETWEEN_RUNS_MILLIS);
                }
            }
        }

        System.out.println("=".repeat(78));
        System.out.printf("Success rate: %d/%d = %.3f%n", successes, SEEDS, ((double) successes) / SEEDS);
        if (successes > 0) {
            System.out.printf("Mean steps-to-success (successful episodes only): %.2f%n",
                    ((double) totalStepsOnSuccess) / successes);
        }
        System.out.printf("Episodes that reached RECOVERING at least once: %d/%d%n", episodesReachingRecovering, SEEDS);
        System.out.printf("Episodes that reached STUCK at least once: %d/%d%n", episodesReachingStuck, SEEDS);
        System.out.println();
        System.out.println("For comparison, the discrete governed_adaptive condition scored 3/3 on");
        System.out.println("this exact task earlier in this project.");
    }

    private static String truncate(String s, int maxLen) {
        if (s == null) {
            return "";
        }
        String oneLine = s.replace("\n", " \\n ");
        return oneLine.length() <= maxLen ? oneLine : oneLine.substring(0, maxLen) + "...";
    }

    private static MockTaskSpec findH005() {
        for (MockTaskSpec t : MockTaskSet.hardTasks()) {
            if (t.taskId().equals("H005_flaky_tool_recovery")) {
                return t;
            }
        }
        throw new IllegalStateException(
                "H005 not found in MockTaskSet.hardTasks() -- has it been renamed?");
    }

    private static CASGovernor buildGovernor(MockTaskSpec task) {
        if (task.isMultiTarget()) {
            return CASGovernor.forExpectedTargets(task.requiredTargets());
        }
        if (task.hasAcceptableTargets()) {
            return CASGovernor.forAnyOfExpectedTargets(task.acceptableTargets());
        }
        return CASGovernor.forExpectedTarget(task.expectedTarget());
    }

    private static CASGovernor buildAdaptiveGovernor(MockTaskSpec task) {
        return buildGovernor(task).withBoundedRetry(
                ADAPTIVE_MAX_TRANSIENT_RETRIES,
                ADAPTIVE_TRANSIENT_ERROR_TYPES
        );
    }

    private static ToolRegistry buildTools(MockTaskSpec task) {
        ToolRegistry tools = new ToolRegistry();
        FileTool fileTool = new FileTool(WORKSPACE.toString());

        if (task.hasFlakyReads()) {
            tools.register(new FlakyFileTool(fileTool, task.flakyReadFailures()));
        } else {
            tools.register(fileTool);
        }

        return tools;
    }

    private static void resetWorkspace(MockTaskSpec task) throws IOException {
        deleteDirectoryIfExists(WORKSPACE);
        Files.createDirectories(WORKSPACE);

        for (var entry : task.files().entrySet()) {
            Path file = WORKSPACE.resolve(entry.getKey()).normalize();

            Path normalizedWorkspace = WORKSPACE.toAbsolutePath().normalize();
            Path normalizedFile = file.toAbsolutePath().normalize();

            if (!normalizedFile.startsWith(normalizedWorkspace)) {
                throw new IOException("Unsafe task file path: " + entry.getKey());
            }

            Files.createDirectories(file.getParent());
            Files.writeString(file, entry.getValue(), StandardCharsets.UTF_8);
        }
    }

    private static void deleteDirectoryIfExists(Path directory) throws IOException {
        if (!Files.exists(directory)) {
            return;
        }

        try (var walk = Files.walk(directory)) {
            walk.sorted(java.util.Comparator.reverseOrder())
                    .forEach(path -> {
                        try {
                            Files.deleteIfExists(path);
                        } catch (IOException e) {
                            throw new RuntimeException("Failed to delete " + path, e);
                        }
                    });
        }
    }
}
