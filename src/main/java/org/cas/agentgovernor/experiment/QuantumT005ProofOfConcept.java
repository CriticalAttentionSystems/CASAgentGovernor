/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package org.cas.agentgovernor.experiment;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;

import org.cas.agentgovernor.core.ToolAction;
import org.cas.agentgovernor.core.ToolResult;
import org.cas.agentgovernor.experiment.tasks.MockTaskSet;
import org.cas.agentgovernor.experiment.tasks.MockTaskSpec;
import org.cas.agentgovernor.tools.FileTool;

import org.iram.omega.iram_omega_q.cognition.Hamiltonian;
import org.iram.omega.iram_omega_q.cognition.QuantumCognitiveState;
import org.iram.omega.iram_omega_q.cognition.QuantumRegulationAgent;
import org.iram.omega.iram_omega_q.control.MuController;

/**
 * Proof of concept: replaces ActionPolicy's ad hoc weighted-choice list with
 * a REAL {@code QuantumRegulationAgent} from the iram_omega_q project on an
 * actual agentic task (T005, the competing-filenames disambiguation task).
 * Every mechanism below is your real machinery, not a discrete re-implementation
 * of it: real {@code Hamiltonian.attentional} locality coupling, real
 * {@code MuController} entropy-driven gain updates, real
 * {@code exp(-i H dt)} unitary evolution, real Born-rule {@code measure()}
 * for action selection.
 *
 * <h2>What this tests</h2>
 * Purely mechanical viability: does wiring the continuous formalism to a real
 * tool-use loop even run, converge, and behave sensibly (entropy/mu tracking
 * task progress in a plausible direction)? It does NOT test whether the
 * continuous formalism's specific quantitative claims (critical ridges,
 * susceptibility peaks) hold on agentic tasks -- that would need a real sweep
 * across (mu0, eta) with proper statistics, not a seed-count pilot on one task.
 *
 * <h2>Content-based grading (closes the earlier index-only gap)</h2>
 * An earlier version of this class declared success purely from
 * {@code candidateIndex == correctIndex} -- i.e. it trusted that measuring
 * the right BASIS INDEX meant the task was solved, without ever checking
 * what was actually read. That's a real, if honest, gap: it doesn't test
 * whether the read content would let anyone downstream produce a correct
 * answer, only whether Born-rule sampling converges on the intended slot.
 * T005's three files genuinely differ in content, not just name:
 * {@code session001_summary.csv} contains {@code status=complete,
 * post_error_burden=3.1}; the "old" decoy contains
 * {@code status=old, post_error_burden=5.2}. This version extracts the
 * expected numeric value directly from {@code task.expectedAnswer()} via
 * regex and checks whether a successfully-read candidate's ACTUAL file
 * content contains it -- success is now grounded in content, the same way
 * {@code AgentRunMetrics.success()} grades the discrete pipeline, not in an
 * index-equality shortcut. For T005 specifically this still agrees with the
 * index check on every episode (the content really is different), but the
 * mechanism no longer assumes that -- it verifies it.
 *
 * <h2>Real, honest simplifications still present -- not hidden, listed explicitly</h2>
 * <ul>
 *   <li><b>Every numeric constant below (coupling, lengthScale, mu bounds,
 *       alpha0/beta0, targetEntropy, disturbance magnitude, energy penalty)
 *       is hand-picked, not fit or validated</b> -- exactly the same caveat
 *       already on record for ActionPolicy.modulate()'s multiplier table.
 *       Nothing here should be read as "the framework predicts X" until
 *       these are actually swept, not guessed.</li>
 *   <li><b>Initial salience is a simple keyword-overlap heuristic</b>
 *       (prompt tokens vs. filename tokens), deliberately NOT using
 *       {@code ToolDiscovery}'s real (stopword-aware) scoring, to keep this
 *       pilot self-contained. It is still non-oracle: the two "session001"
 *       files score identically by design, so the initial state cannot
 *       cheat past T005's actual difficulty.</li>
 * </ul>
 *
 * <h2>How disturbance is injected</h2>
 * {@code QuantumRegulationAgent.setEmotionalNoise(double)} sets a
 * PERSISTENT field, read by {@code ExternalNoiseModel} on every subsequent
 * {@code step()} until changed again -- not a one-shot pulse. This class
 * explicitly resets it to 0 each step and only raises it for exactly one
 * following step after a wrong (but successfully read) guess, so a single
 * wrong guess produces one disturbance pulse, not a permanently elevated
 * noise floor.
 *
 * <h2>How "wrong guess" feeds back into the Hamiltonian</h2>
 * {@code Hamiltonian.attentional(energies, coupling, lengthScale)} closes
 * over the {@code energies} array BY REFERENCE. Mutating that array after
 * construction (raising a confirmed-wrong candidate's diagonal energy)
 * really does change what the Hamiltonian returns on every subsequent
 * {@code step()} -- a genuine energetic penalty via the real mechanism, not
 * a simulated one bolted on separately.
 *
 * @author veronique
 */
public class QuantumT005ProofOfConcept {

    private static final Path WORKSPACE = Path.of("test_workspace_quantum_poc");
    private static final int MAX_STEPS = 8;
    private static final int SEEDS = 200; // bumped from the original 30-seed pilot for a tighter estimate

    // ---- Hand-picked, unvalidated constants -- see class doc. ----
    private static final double HAMILTONIAN_COUPLING = 0.6;
    private static final double HAMILTONIAN_LENGTH_SCALE = 1.5;
    private static final double MU_INIT = 0.2;
    private static final double MU_MIN = 0.0;
    private static final double MU_MAX = 0.95;
    private static final double MU_ALPHA0 = 0.4;
    private static final double MU_BETA0 = 0.6;
    private static final double MU_TARGET_ENTROPY = 0.3;
    private static final double DISTURBANCE_ON_WRONG_GUESS = 0.35;
    private static final double WRONG_GUESS_ENERGY_PENALTY = 1.5;
    private static final double INITIAL_SALIENCE_PHASE_NOISE = 0.05;

    /** Matches the last decimal number in a string, e.g. extracts "3.1" from expectedAnswer. */
    private static final java.util.regex.Pattern NUMERIC_VALUE_PATTERN =
            java.util.regex.Pattern.compile("(\\d+\\.\\d+)");

    public static void main(String[] args) throws Exception {
        MockTaskSpec task = findT005();

        System.out.println("Quantum T005 proof of concept -- " + SEEDS
                + " seeds, max " + MAX_STEPS + " steps each.");
        System.out.println("Candidates (basis order): " + new ArrayList<>(task.files().keySet()).stream()
                .sorted().toList());
        System.out.println("Correct target: " + task.expectedTarget());
        System.out.println("=".repeat(78));

        int successes = 0;
        long totalStepsOnSuccess = 0;

        for (int seed = 0; seed < SEEDS; seed++) {
            EpisodeResult result = runOneEpisode(task, seed);

            if (result.success()) {
                successes++;
                totalStepsOnSuccess += result.stepsTaken();
            }

            System.out.printf(
                    "seed %2d: %-7s steps=%d finalEntropy=%.4f finalMu=%.4f%n",
                    seed,
                    result.success() ? "SUCCESS" : "FAIL",
                    result.stepsTaken(),
                    result.finalEntropy(),
                    result.finalMu()
            );
        }

        System.out.println("=".repeat(78));
        System.out.printf("Success rate: %d/%d = %.3f%n", successes, SEEDS, ((double) successes) / SEEDS);
        if (successes > 0) {
            System.out.printf("Mean steps-to-success (successful episodes only): %.2f%n",
                    ((double) totalStepsOnSuccess) / successes);
        }
        System.out.println();
        System.out.println("Reminder: this tests mechanical viability and index-selection");
        System.out.println("convergence only -- see class doc for what it does NOT test.");
    }

    private static MockTaskSpec findT005() {
        for (MockTaskSpec t : MockTaskSet.fileRecoveryTasks()) {
            if (t.taskId().equals("T005_similar_competing_filenames")) {
                return t;
            }
        }
        throw new IllegalStateException(
                "T005 not found in MockTaskSet.fileRecoveryTasks() -- has it been renamed?");
    }

    private static EpisodeResult runOneEpisode(MockTaskSpec task, int seed) throws IOException {
        resetWorkspace(task);

        List<String> candidatePaths = new ArrayList<>(task.files().keySet());
        candidatePaths.sort(String::compareTo); // stable, deterministic basis ordering

        int dim = candidatePaths.size();
        int correctIndex = candidatePaths.indexOf(task.expectedTarget());
        if (correctIndex < 0) {
            throw new IllegalStateException(
                    "expectedTarget not found among task.files() keys: " + task.expectedTarget());
        }

        Random rng = new Random(seed);

        String expectedValue = extractNumericValue(task.expectedAnswer());

        double[] salience = new double[dim];
        for (int i = 0; i < dim; i++) {
            salience[i] = keywordOverlapScore(task.prompt(), candidatePaths.get(i));
        }

        QuantumCognitiveState psi =
                QuantumCognitiveState.salienceCoherent(salience, INITIAL_SALIENCE_PHASE_NOISE, rng);

        // Mutable -- see class doc on how wrong-guess feedback uses this.
        double[] energies = new double[dim];
        Hamiltonian hamiltonian = Hamiltonian.attentional(energies, HAMILTONIAN_COUPLING, HAMILTONIAN_LENGTH_SCALE);

        MuController muController = new MuController(
                MU_INIT, MU_MIN, MU_MAX, MU_ALPHA0, MU_BETA0, MU_TARGET_ENTROPY
        );

        QuantumRegulationAgent agent = new QuantumRegulationAgent(psi, hamiltonian, muController, rng);
        agent.setControlOrdering(QuantumRegulationAgent.ControlOrdering.REGULATION_FIRST);

        FileTool fileTool = new FileTool(WORKSPACE.toString());

        boolean[] confirmedWrong = new boolean[dim];
        double pendingNoise = 0.0;

        for (int step = 0; step < MAX_STEPS; step++) {
            // Apply and immediately consume any noise queued by the PREVIOUS
            // step's outcome -- see class doc: setEmotionalNoise is
            // persistent, so this must be reset every step or a single
            // wrong guess would disturb every future step too.
            agent.setEmotionalNoise(pendingNoise);
            pendingNoise = 0.0;

            agent.step();

            int candidateIndex = agent.measure();
            String candidatePath = candidatePaths.get(candidateIndex);

            ToolAction action = new ToolAction("file", "READ_FILE", candidatePath, "");
            ToolResult result = fileTool.execute(action);

            boolean contentMatches = result.success()
                    && result.output() != null
                    && result.output().contains(expectedValue);

            if (contentMatches) {
                return new EpisodeResult(
                        true,
                        step + 1,
                        QuantumCognitiveState.vonNeumannEntropy(agent.state()),
                        agent.mindfulnessLevel()
                );
            }

            // Either a real tool failure, or a successful-but-wrong-content
            // read (T005's actual difficulty: the "old" decoy reads
            // successfully, it just contains different data). Both are
            // treated as a disturbance: queue noise for next step, and if
            // this is a NEWLY confirmed-wrong candidate, permanently raise
            // its Hamiltonian energy via the real, mutable-array mechanism
            // described in the class doc.
            if (result.success() && !confirmedWrong[candidateIndex]) {
                confirmedWrong[candidateIndex] = true;
                energies[candidateIndex] += WRONG_GUESS_ENERGY_PENALTY;
            }
            pendingNoise = DISTURBANCE_ON_WRONG_GUESS;
        }

        return new EpisodeResult(
                false,
                MAX_STEPS,
                QuantumCognitiveState.vonNeumannEntropy(agent.state()),
                agent.mindfulnessLevel()
        );
    }

    /**
     * Simple, honest (non-oracle) keyword-overlap salience: shared tokens
     * between the prompt and a candidate filename, with a +1 floor so every
     * candidate starts with nonzero salience. Deliberately simpler than
     * ToolDiscovery's real stopword-aware scoring -- see class doc.
     */
    /**
     * Extracts the last decimal number appearing in a string, e.g. "3.1"
     * from "Session 001 completed successfully with post-error burden 3.1."
     * Used to ground success grading in actual file content rather than a
     * basis-index shortcut -- see class doc.
     */
    private static String extractNumericValue(String expectedAnswer) {
        java.util.regex.Matcher matcher = NUMERIC_VALUE_PATTERN.matcher(expectedAnswer);
        String last = null;
        while (matcher.find()) {
            last = matcher.group(1);
        }
        if (last == null) {
            throw new IllegalStateException(
                    "No decimal number found in expectedAnswer to grade against: " + expectedAnswer);
        }
        return last;
    }

    private static double keywordOverlapScore(String prompt, String candidatePath) {
        Set<String> promptTokens = tokenize(prompt);
        Set<String> fileTokens = tokenize(candidatePath);

        long overlap = fileTokens.stream().filter(promptTokens::contains).count();
        return 1.0 + overlap;
    }

    private static Set<String> tokenize(String s) {
        Set<String> tokens = new HashSet<>();
        for (String raw : s.toLowerCase().split("[^a-z0-9]+")) {
            if (raw.length() >= 3) {
                tokens.add(raw);
            }
        }
        return tokens;
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

    private record EpisodeResult(
            boolean success,
            int stepsTaken,
            double finalEntropy,
            double finalMu
    ) {
    }
}
