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
import java.util.List;
import java.util.Random;

import org.cas.agentgovernor.agents.InducedCostStochasticToolAgent;
import org.cas.agentgovernor.agents.InducedCostStochasticToolAgent.BackoffMode;
import org.cas.agentgovernor.agents.InducedCostStochasticToolAgent.CostMechanism;
import org.cas.agentgovernor.core.AgentEvent;
import org.cas.agentgovernor.core.CASGovernor;
import org.cas.agentgovernor.experiment.tasks.MockTaskSet;
import org.cas.agentgovernor.experiment.tasks.MockTaskSpec;
import org.cas.agentgovernor.metrics.AgentRunMetrics;
import org.cas.agentgovernor.policy.ActionPolicy;
import org.cas.agentgovernor.policy.PolicyParameters;
import org.cas.agentgovernor.tools.FileTool;
import org.cas.agentgovernor.tools.ToolRegistry;

/**
 * Reviewer-hardening experiments for the control-generated-disturbance study,
 * all on the free stochastic-policy substrate. Produces two DEMONSTRATED
 * results that do not need any LLM quota:
 *
 * <ol>
 *   <li><b>Mechanism robustness.</b> The induced-cost feedback loop is shown
 *   under TWO distinct cost channels -- RESULT_INJECTION (force the next tool
 *   call to fail) and ARGUMENT_CORRUPTION (jostle the action target and let the
 *   real tool decide the outcome) -- so the collapse is not an artifact of the
 *   specific injection. Cost-blind amplification and cost-aware suppression
 *   should hold qualitatively under both.</li>
 *
 *   <li><b>Baseline ladder + weak-governor control.</b> Compares no-governor,
 *   an ordinary governor at alpha=0, cost-blind, cost-aware, and a FIXED
 *   weak-governor that intervenes at a constant probability matched to
 *   cost-aware's realized intervention rate. This tests whether cost-aware
 *   backoff is genuinely ADAPTIVE or merely equivalent to intervening less
 *   often.</li>
 * </ol>
 *
 * Writes output/induced_robustness_mechanisms.csv and
 * output/induced_robustness_baselines.csv.
 *
 * @author veronique
 */
public class RunInducedCostRobustness {

    private static final int SEEDS = 2000;
    private static final int MAX_STEPS = 16;
    private static final double[] ALPHAS = {0.0, 1.0, 2.0};
    private static final double BACKOFF_LAMBDA = 0.3;
    private static final double BACKOFF_K = 3.0;
    private static final Path WORKSPACE_ROOT = Path.of("test_workspace_robustness");

    private record AgentType(String name, PolicyParameters parameters) {
    }

    /** One experimental condition. */
    private record Cell(
            String label, boolean reweight, boolean enforceBlocking, double alpha,
            BackoffMode backoffMode, double k, double fixedP, CostMechanism mechanism
    ) {
    }

    /** Aggregate over seeds. */
    private record Stats(
            String agent, String label, double alpha, String mechanism, String backoff,
            int n, double success, double meanG, double meanInduced, double meanHardBlocks,
            double meanBurden, double meanApplyRate
    ) {
    }

    public static void main(String[] args) throws Exception {
        List<AgentType> agents = List.of(
                new AgentType("impulsive", PolicyParameters.impulsive()),
                new AgentType("persistent", PolicyParameters.persistent())
        );
        Files.createDirectories(Path.of("output"));

        // ---- Part 1: mechanism robustness ----
        List<Stats> mechStats = new ArrayList<>();
        for (AgentType agent : agents) {
            for (CostMechanism mech : CostMechanism.values()) {
                for (double alpha : ALPHAS) {
                    mechStats.add(runCell(agent, new Cell(
                            "blind", true, true, alpha, BackoffMode.NONE, 0, -1, mech)));
                    mechStats.add(runCell(agent, new Cell(
                            "aware_k3.0", true, true, alpha, BackoffMode.ADAPTIVE, BACKOFF_K, -1, mech)));
                }
            }
        }
        writeCsv(Path.of("output/induced_robustness_mechanisms.csv"), mechStats);
        System.out.println("Wrote: output/induced_robustness_mechanisms.csv");

        // ---- Part 2: baseline ladder + fixed-weak control (RESULT_INJECTION, alpha=2) ----
        double alpha2 = 2.0;
        List<Stats> baseStats = new ArrayList<>();
        for (AgentType agent : agents) {
            Stats ungoverned = runCell(agent, new Cell(
                    "ungoverned", false, false, 0.0, BackoffMode.NONE, 0, -1, CostMechanism.RESULT_INJECTION));
            Stats governedA0 = runCell(agent, new Cell(
                    "governor_alpha0", true, true, 0.0, BackoffMode.NONE, 0, -1, CostMechanism.RESULT_INJECTION));
            Stats costBlind = runCell(agent, new Cell(
                    "cost_blind_alpha2", true, true, alpha2, BackoffMode.NONE, 0, -1, CostMechanism.RESULT_INJECTION));
            Stats costAware = runCell(agent, new Cell(
                    "cost_aware_alpha2", true, true, alpha2, BackoffMode.ADAPTIVE, BACKOFF_K, -1, CostMechanism.RESULT_INJECTION));

            // Weak-governor control: FIXED apply probability matched to
            // cost-aware's realized intervention rate, so any difference is
            // attributable to ADAPTIVITY (intervening when it matters) rather
            // than to intervening less on average.
            double matchedP = costAware.meanApplyRate();
            Stats fixedWeak = runCell(agent, new Cell(
                    "fixed_weak_alpha2", true, true, alpha2, BackoffMode.FIXED, 0, matchedP, CostMechanism.RESULT_INJECTION));

            baseStats.add(ungoverned);
            baseStats.add(governedA0);
            baseStats.add(costBlind);
            baseStats.add(costAware);
            baseStats.add(fixedWeak);
            System.out.printf("  %s: cost-aware realized apply-rate = %.3f -> fixed-weak matched to it%n",
                    agent.name(), matchedP);
        }
        writeCsv(Path.of("output/induced_robustness_baselines.csv"), baseStats);
        System.out.println("Wrote: output/induced_robustness_baselines.csv");

        deleteDirectoryIfExists(WORKSPACE_ROOT);
        printSummary(mechStats, baseStats);
    }

    private static Stats runCell(AgentType agent, Cell cell) throws IOException {
        List<MockTaskSpec> tasks = MockTaskSet.fileRecoveryTasks();
        Path workspace = WORKSPACE_ROOT.resolve(agent.name() + "_" + cell.label() + "_" + cell.mechanism()
                + "_a" + String.format("%.1f", cell.alpha()));

        double successSum = 0, gSum = 0, inducedSum = 0, hbSum = 0, burdenSum = 0, applySum = 0;
        int n = 0;
        try {
            for (MockTaskSpec task : tasks) {
                for (int seed = 0; seed < SEEDS; seed++) {
                    long ps = seedFor(seed, task.taskId(), agent.name(), cell.label(), cell.mechanism().name(), "policy");
                    long is = seedFor(seed, task.taskId(), agent.name(), cell.label(), cell.mechanism().name(), "induced");
                    resetWorkspace(task, workspace);

                    ToolRegistry tools = new ToolRegistry();
                    tools.register(new FileTool(workspace.toString()));
                    ActionPolicy policy = new ActionPolicy(agent.parameters(), new Random(ps),
                            task.expectedTarget(), task.expectedAnswer());
                    CASGovernor governor = CASGovernor.forExpectedTarget(task.expectedTarget());

                    InducedCostStochasticToolAgent a = new InducedCostStochasticToolAgent(
                            policy, tools, governor, cell.reweight(), cell.enforceBlocking(), 0,
                            cell.alpha(), new Random(is), cell.backoffMode(), BACKOFF_LAMBDA, cell.k(),
                            cell.fixedP(), cell.mechanism());
                    a.run(task.prompt(), MAX_STEPS);

                    AgentRunMetrics m = AgentRunMetrics.fromHistory("x", a.history(), task.expectedAnswer());
                    successSum += m.success() ? 1 : 0;
                    burdenSum += m.postErrorToolBurden();
                    gSum += mean(a.regulationIntensityHistory());
                    inducedSum += a.inducedFailureCount();
                    hbSum += countHardBlocks(a.history());
                    int applied = a.regulationActiveCount();
                    int backed = a.backoffStepCount();
                    applySum += (applied + backed) == 0 ? 1.0 : (double) applied / (applied + backed);
                    n++;
                }
            }
        } finally {
            deleteDirectoryIfExists(workspace);
        }
        return new Stats(agent.name(), cell.label(), cell.alpha(), cell.mechanism().name(),
                cell.backoffMode().name(), n, successSum / n, gSum / n, inducedSum / n, hbSum / n,
                burdenSum / n, applySum / n);
    }

    private static void printSummary(List<Stats> mech, List<Stats> base) {
        System.out.println("\n==== Mechanism robustness (success | mean_g by mechanism x backoff x alpha) ====");
        for (String agent : List.of("impulsive", "persistent")) {
            System.out.println("-- " + agent + " --");
            for (String mm : List.of("RESULT_INJECTION", "ARGUMENT_CORRUPTION")) {
                for (String bo : List.of("NONE", "ADAPTIVE")) {
                    StringBuilder sb = new StringBuilder(String.format("  %-18s %-8s ", mm, bo));
                    for (double a : ALPHAS) {
                        Stats s = find(mech, agent, mm, bo, a);
                        if (s != null) {
                            sb.append(String.format("a%.0f: succ=%.3f g=%.3f  ", a, s.success(), s.meanG()));
                        }
                    }
                    System.out.println(sb);
                }
            }
        }
        System.out.println("\n==== Baseline ladder at alpha=2 (RESULT_INJECTION) ====");
        for (String agent : List.of("impulsive", "persistent")) {
            System.out.println("-- " + agent + " --");
            for (Stats s : base) {
                if (!s.agent().equals(agent)) continue;
                System.out.printf("  %-20s succ=%.3f  g=%.3f  hard_blocks=%.2f  apply_rate=%.3f%n",
                        s.label(), s.success(), s.meanG(), s.meanHardBlocks(), s.meanApplyRate());
            }
        }
    }

    private static Stats find(List<Stats> l, String agent, String mech, String backoff, double alpha) {
        for (Stats s : l) {
            if (s.agent().equals(agent) && s.mechanism().equals(mech)
                    && s.backoff().equals(backoff) && Math.abs(s.alpha() - alpha) < 1e-9) {
                return s;
            }
        }
        return null;
    }

    private static void writeCsv(Path path, List<Stats> stats) throws IOException {
        List<String> rows = new ArrayList<>();
        rows.add("agent,label,alpha,mechanism,backoff,n,success_rate,mean_g,mean_induced,"
                + "mean_hard_blocks,mean_burden,mean_apply_rate");
        for (Stats s : stats) {
            rows.add(String.join(",", s.agent(), s.label(), String.format("%.1f", s.alpha()),
                    s.mechanism(), s.backoff(), Integer.toString(s.n()),
                    f(s.success()), f(s.meanG()), f(s.meanInduced()), f(s.meanHardBlocks()),
                    f(s.meanBurden()), f(s.meanApplyRate())));
        }
        Files.write(path, rows, StandardCharsets.UTF_8);
    }

    private static String f(double v) {
        return String.format("%.5f", v);
    }

    private static int countHardBlocks(List<AgentEvent> history) {
        int c = 0;
        for (AgentEvent e : history) {
            if (e.message() != null && e.message().startsWith("HARD_BLOCK:")) {
                c++;
            }
        }
        return c;
    }

    private static double mean(List<Double> xs) {
        if (xs.isEmpty()) {
            return 0.0;
        }
        double s = 0;
        for (double x : xs) {
            s += x;
        }
        return s / xs.size();
    }

    private static long seedFor(int seed, String taskId, String agent, String label, String mech, String stream) {
        long h = 1125899906842597L;
        h = 31 * h + seed;
        h = 31 * h + taskId.hashCode();
        h = 31 * h + agent.hashCode();
        h = 31 * h + label.hashCode();
        h = 31 * h + mech.hashCode();
        h = 31 * h + stream.hashCode();
        // splitmix64 finalize so streams differing by one field decorrelate.
        h = (h ^ (h >>> 30)) * 0xbf58476d1ce4e5b9L;
        h = (h ^ (h >>> 27)) * 0x94d049bb133111ebL;
        return h ^ (h >>> 31);
    }

    private static void resetWorkspace(MockTaskSpec task, Path workspace) throws IOException {
        deleteDirectoryIfExists(workspace);
        Files.createDirectories(workspace);
        for (var e : task.files().entrySet()) {
            Path file = workspace.resolve(e.getKey()).normalize();
            if (!file.toAbsolutePath().normalize().startsWith(workspace.toAbsolutePath().normalize())) {
                throw new IOException("Unsafe task file path: " + e.getKey());
            }
            Files.createDirectories(file.getParent());
            Files.writeString(file, e.getValue(), StandardCharsets.UTF_8);
        }
        Files.createDirectories(workspace.resolve("examples"));
    }

    private static void deleteDirectoryIfExists(Path directory) throws IOException {
        if (!Files.exists(directory)) {
            return;
        }
        try (var walk = Files.walk(directory)) {
            walk.sorted(java.util.Comparator.reverseOrder()).forEach(p -> {
                try {
                    Files.deleteIfExists(p);
                } catch (IOException e) {
                    throw new RuntimeException("Failed to delete " + p, e);
                }
            });
        }
    }
}
