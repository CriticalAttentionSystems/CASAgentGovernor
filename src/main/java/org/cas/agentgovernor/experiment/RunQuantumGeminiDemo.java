/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package org.cas.agentgovernor.experiment;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

import org.cas.agentgovernor.agents.QuantumGovernedLLMAgent;
import org.cas.agentgovernor.core.CASGovernor;
import org.cas.agentgovernor.experiment.tasks.MockTaskSet;
import org.cas.agentgovernor.experiment.tasks.MockTaskSpec;
import org.cas.agentgovernor.llm.GeminiLLMClient;
import org.cas.agentgovernor.metrics.AgentRunMetrics;
import org.cas.agentgovernor.tools.FileTool;
import org.cas.agentgovernor.tools.FlakyFileTool;
import org.cas.agentgovernor.tools.ToolRegistry;

/**
 * Single-episode, cheap smoke test for {@link QuantumGovernedLLMAgent},
 * mirroring {@code RunGeminiDemo}'s pattern -- one real Gemini episode,
 * printed transcript and metrics, no checkpointing.
 *
 * <p>Uses H005 (flaky_tool_recovery), not T005. This is a deliberate choice,
 * not an arbitrary pick between the two hard tasks offered: H005's
 * disturbance is GUARANTEED, not merely probable. Its primary file fails
 * deterministically on its first {@code flakyReadFailures} attempts
 * regardless of which candidate the LLM tries first -- unlike H001, where a
 * real Gemini run already showed governed/ungoverned solving it 3/3 with no
 * wrong turns at all under normal conditions.
 *
 * <h2>Uses the ADAPTIVE (bounded-retry) governor, not the plain one</h2>
 * The first version of this demo used {@code buildGovernor()} (strict,
 * unconditional repeat-blocking), which reproduced the exact same known
 * H005 liability already diagnosed and fixed on the discrete pipeline:
 * one real transient failure, then every retry attempt hard-blocked for the
 * rest of the episode, burning the step budget on repeated blocks instead
 * of finding the backup. That is not a new discovery -- it is the same
 * mechanism, confirmed to generalize to a real LLM, using a governor
 * variant already known to have this problem. This version uses
 * {@code buildAdaptiveGovernor()} instead (bounded retry on
 * TRANSIENT_READ_ERROR failures specifically), matching what already fixed
 * this exact liability 0/3 to 3/3 on the discrete H005 comparison.
 *
 * <p>Governor and tool construction otherwise mirror
 * {@code RunGeminiHardTaskComparison} exactly -- see that class for the
 * full rationale on each.
 *
 * @author veronique
 */
public class RunQuantumGeminiDemo {

    private static final Path WORKSPACE = Path.of("test_workspace_quantum_gemini_demo");
    private static final int MAX_STEPS = 8; // matches RunGeminiHardTaskComparison's H005 budget

    // Matches RunGeminiHardTaskComparison's ADAPTIVE_* constants exactly.
    private static final int ADAPTIVE_MAX_TRANSIENT_RETRIES = 2;
    private static final java.util.Set<String> ADAPTIVE_TRANSIENT_ERROR_TYPES =
            java.util.Set.of("TRANSIENT_READ_ERROR");

    public static void main(String[] args) throws Exception {
        MockTaskSpec task = findH005();
        resetWorkspace(task);

        ToolRegistry tools = buildTools(task);
        CASGovernor governor = buildAdaptiveGovernor(task);

        String answer;
        AgentRunMetrics metrics;

        try (GeminiLLMClient llm = new GeminiLLMClient("gemini-2.5-flash")) {
            QuantumGovernedLLMAgent agent = new QuantumGovernedLLMAgent(
                    llm,
                    governor,
                    tools,
                    8_000L,
                    /* quantumSeed = */ 42L
            );

            answer = agent.run(task.prompt(), MAX_STEPS);

            System.out.println("\n=== FINAL ANSWER ===");
            System.out.println(answer);

            System.out.println("\n=== EXPECTED ANSWER ===");
            System.out.println(task.expectedAnswer());

            System.out.println("\n=== HISTORY (observational signal -- see class doc, NOT what drove the LLM) ===");
            agent.history().forEach(System.out::println);

            System.out.println("\n=== ENTROPY / MU / CONTROL-SIGNAL TRACE (what actually drove the LLM) ===");
            for (int i = 0; i < agent.entropyHistory().size(); i++) {
                System.out.printf("step %d: entropy=%.4f mu=%.4f signal=%s%n",
                        i, agent.entropyHistory().get(i), agent.muHistory().get(i),
                        agent.controlSignalHistory().get(i).state());
            }

            metrics = AgentRunMetrics.fromHistory("quantum_governed_gemini_h005", agent.history(), task.expectedAnswer());
        }

        System.out.println("\n=== METRICS ===");
        System.out.println(metrics.toCsvHeader());
        System.out.println(metrics.toCsvRow());
        System.out.println(metrics);
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

    /** Mirrors RunGeminiHardTaskComparison.buildGovernor() exactly. */
    private static CASGovernor buildGovernor(MockTaskSpec task) {
        if (task.isMultiTarget()) {
            return CASGovernor.forExpectedTargets(task.requiredTargets());
        }
        if (task.hasAcceptableTargets()) {
            return CASGovernor.forAnyOfExpectedTargets(task.acceptableTargets());
        }
        return CASGovernor.forExpectedTarget(task.expectedTarget());
    }

    /** Mirrors RunGeminiHardTaskComparison.buildAdaptiveGovernor() exactly -- see class doc for why this is used instead of buildGovernor(). */
    private static CASGovernor buildAdaptiveGovernor(MockTaskSpec task) {
        return buildGovernor(task).withBoundedRetry(
                ADAPTIVE_MAX_TRANSIENT_RETRIES,
                ADAPTIVE_TRANSIENT_ERROR_TYPES
        );
    }

    /** Mirrors RunGeminiHardTaskComparison.buildTools() exactly -- see that class for the full rationale. */
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
