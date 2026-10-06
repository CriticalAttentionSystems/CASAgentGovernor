/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package org.cas.agentgovernor.experiment.tasks;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 *
 * @author veronique
 */

public class MockTaskSet {

    public static List<MockTaskSpec> fileRecoveryTasks() {
        List<MockTaskSpec> tasks = new ArrayList<>();

        tasks.add(new MockTaskSpec(
                "T001_wrong_filename",
                """
                Find the trial config file and summarize the baseline,
                disturbance, and recovery phases.

                The config is probably examples/trial_config.json.
                """,
                "configs/trial_config_demo.json",
                "The trial config defines three phases: baseline from 0 to 120 seconds, disturbance from 120 to 240 seconds, and recovery from 240 to 420 seconds.",
                files(Map.of(
                        "configs/trial_config_demo.json",
                        """
                        {
                          "trial_id": "session001_trial1",
                          "condition": "normal_recovery",
                          "baseline": { "start_sec": 0, "end_sec": 120 },
                          "disturbance": { "start_sec": 120, "end_sec": 240 },
                          "recovery": { "start_sec": 240, "end_sec": 420 }
                        }
                        """
                ))
        ));

        tasks.add(new MockTaskSpec(
                "T002_wrong_folder",
                """
                Read output/summary.csv and report the mean post-error burden.

                The summary is probably at output/summary.csv.
                """,
                "output/demo/summary.csv",
                "The mean post-error burden is 3.7.",
                files(Map.of(
                        "output/demo/summary.csv",
                        """
                        run_id,mean_post_error_burden,success_rate
                        demo001,3.7,1.0
                        """
                ))
        ));

        tasks.add(new MockTaskSpec(
                "T003_wrong_extension",
                """
                Read configs/session_config.json and summarize the condition.

                The file may have been saved under a slightly different extension.
                """,
                "configs/session_config.yaml",
                "The condition is breath_recovery.",
                files(Map.of(
                        "configs/session_config.yaml",
                        """
                        trial_id: session002_trial1
                        condition: breath_recovery
                        baseline_sec: 120
                        disturbance_sec: 120
                        recovery_sec: 300
                        """
                ))
        ));

        tasks.add(new MockTaskSpec(
                "T004_empty_expected_directory",
                """
                Read examples/run_summary.csv and report whether the run succeeded.

                If examples is empty, inspect the workspace.
                """,
                "results/run_summary.csv",
                "The run succeeded.",
                files(Map.of(
                        "results/run_summary.csv",
                        """
                        run_id,status,post_error_burden
                        run004,succeeded,2.9
                        """
                ))
        ));

        tasks.add(new MockTaskSpec(
                "T005_similar_competing_filenames",
                """
                Read the session 001 summary.

                Use the current session001 summary, not old or unrelated files.
                """,
                "output/session001_summary.csv",
                "Session 001 completed successfully with post-error burden 3.1.",
                files(Map.of(
                        "output/session000_summary.csv",
                        """
                        session,status,post_error_burden
                        session000,complete,4.8
                        """,
                        "output/session001_summary.csv",
                        """
                        session,status,post_error_burden
                        session001,complete,3.1
                        """,
                        "output/session001_summary_old.csv",
                        """
                        session,status,post_error_burden
                        session001,old,5.2
                        """
                ))
        ));

        tasks.add(new MockTaskSpec(
                "T006_old_filename_mentioned",
                """
                The old notes say the file is config.json.
                Find the current config and summarize the phases.
                """,
                "configs/trial_config_current.json",
                "The current config defines baseline, disturbance, and recovery phases.",
                files(Map.of(
                        "configs/config_legacy.json",
                        """
                        {
                          "version": "legacy",
                          "note": "old config, do not use"
                        }
                        """,
                        "configs/trial_config_current.json",
                        """
                        {
                          "version": "current",
                          "phases": ["baseline", "disturbance", "recovery"]
                        }
                        """
                ))
        ));

        tasks.add(new MockTaskSpec(
                "T007_nested_directory",
                """
                Find the final metrics file and report whether the run succeeded.

                The file is probably metrics.csv.
                """,
                "runs/session001/final/metrics.csv",
                "The run succeeded.",
                files(Map.of(
                        "runs/session001/final/metrics.csv",
                        """
                        run_id,succeeded,post_error_burden
                        session001,true,2.4
                        """
                ))
        ));

        tasks.add(new MockTaskSpec(
                "T008_latest_summary",
                """
                Read the latest summary file and report the post-error burden.

                There may be multiple dated summary files.
                """,
                "output/summary_2026_07_02.csv",
                "The latest post-error burden is 2.6.",
                files(Map.of(
                        "output/summary_2026_07_01.csv",
                        """
                        date,post_error_burden
                        2026-07-01,4.0
                        """,
                        "output/summary_2026_07_02.csv",
                        """
                        date,post_error_burden
                        2026-07-02,2.6
                        """
                ))
        ));

        return tasks;
    }

    /**
     * A second, harder task tier, added after the original 8 tasks turned
     * out to be too easy for a frontier LLM (Gemini 2.5 Flash solved 100%
     * of ungoverned AND governed episodes across all 8 -- a ceiling effect
     * that made governed-vs-ungoverned comparisons on real Gemini data
     * completely uninformative, even though the sham_regulation comparison
     * still showed a large, real gap). fileRecoveryTasks() above is left
     * untouched deliberately -- it already has real, collected data behind
     * it and reproducibility matters more than tidiness here.
     *
     * Each task targets a genuinely different source of difficulty for a
     * CAPABLE model (not just "harder for a weighted random sampler"):
     *
     *   H001 - content-based disambiguation among several structurally
     *          identical candidates (filenames alone don't distinguish the
     *          right one; the agent has to read multiple candidates and
     *          reason about their content).
     *   H002 - multi-file synthesis: no single file contains the answer;
     *          two files must be read and combined (here, arithmetically).
     *   H003 - a deceptive/stale first-found file: the naive first guess
     *          resolves successfully and LOOKS like an answer, but its own
     *          content flags it as deprecated/invalid, requiring the agent
     *          to notice that and keep looking rather than trusting the
     *          first successful read.
     *   H004 - the same content-based-filtering idea as H001 but at larger
     *          scale (5 candidates instead of 3), raising the cost of
     *          exhaustively checking every candidate rather than guessing.
     *
     * Pair this task set with a materially tighter step budget than the
     * original 16 (see RunGeminiHardTaskComparison, which uses 8) --
     * budget pressure was the other lever identified as untested; a capable
     * model with a generous budget may simply brute-force through even
     * these harder tasks, which would again produce an uninformative
     * ceiling effect.
     */
    public static List<MockTaskSpec> hardTasks() {
        List<MockTaskSpec> tasks = new ArrayList<>();

        tasks.add(new MockTaskSpec(
                "H001_content_disambiguation",
                """
                Find the metrics file for the run under runs/session_beta
                that actually succeeded, and report its post-error burden.

                It's probably at runs/session_beta/attempt1/metrics.csv.
                """,
                "runs/session_beta/attempt3/metrics.csv",
                "The run succeeded with a post-error burden of 1.8.",
                files(Map.of(
                        "runs/session_alpha/attempt1/metrics.csv",
                        """
                        run_id,succeeded,post_error_burden
                        alpha_attempt1,false,6.2
                        """,
                        "runs/session_beta/attempt1/metrics.csv",
                        """
                        run_id,succeeded,post_error_burden
                        attempt1,false,9.1
                        """,
                        "runs/session_beta/attempt2/metrics.csv",
                        """
                        run_id,succeeded,post_error_burden
                        attempt2,false,7.4
                        """,
                        "runs/session_beta/attempt3/metrics.csv",
                        """
                        run_id,succeeded,post_error_burden
                        attempt3,true,1.8
                        """
                ))
        ));

        tasks.add(new MockTaskSpec(
                "H002_multi_file_synthesis",
                """
                Read data/raw_metrics.csv and data/calibration.json, then
                report the corrected post-error burden: the raw burden
                multiplied by the correction factor.
                """,
                "data/raw_metrics.csv",
                "The corrected post-error burden is 3.0.",
                files(Map.of(
                        "data/raw_metrics.csv",
                        """
                        run_id,raw_burden
                        run010,5.0
                        """,
                        "data/calibration.json",
                        """
                        {
                          "correction_factor": 0.6,
                          "note": "apply to raw_burden before reporting"
                        }
                        """
                )),
                Set.of("data/raw_metrics.csv", "data/calibration.json")
        ));

        tasks.add(new MockTaskSpec(
                "H003_deprecated_file_trap",
                """
                Read config/settings.json and report the condition.
                """,
                "config/settings_v2.json",
                "The condition is thermal_shock.",
                files(Map.of(
                        "config/settings.json",
                        """
                        {
                          "status": "DEPRECATED - see settings_v2.json",
                          "condition": "unknown"
                        }
                        """,
                        "config/settings_v2.json",
                        """
                        {
                          "status": "active",
                          "condition": "thermal_shock"
                        }
                        """
                ))
        ));

        tasks.add(new MockTaskSpec(
                "H004_large_scale_filtering",
                """
                Several run logs exist under logs/. Find the one run that
                actually completed (not aborted) and report its
                post-error burden.
                """,
                "logs/run_2026_07_02b.csv",
                "The post-error burden is 2.2.",
                files(Map.of(
                        "logs/run_2026_07_01a.csv",
                        """
                        run_id,status,post_error_burden
                        run_2026_07_01a,aborted,8.8
                        """,
                        "logs/run_2026_07_01b.csv",
                        """
                        run_id,status,post_error_burden
                        run_2026_07_01b,aborted,7.5
                        """,
                        "logs/run_2026_07_02a.csv",
                        """
                        run_id,status,post_error_burden
                        run_2026_07_02a,aborted,6.9
                        """,
                        "logs/run_2026_07_02b.csv",
                        """
                        run_id,status,post_error_burden
                        run_2026_07_02b,complete,2.2
                        """,
                        "logs/run_2026_07_03a.csv",
                        """
                        run_id,status,post_error_burden
                        run_2026_07_03a,aborted,9.4
                        """
                ))
        ));

        // H005 is deliberately NOT designed to make governed win -- see
        // FlakyFileTool's class doc. data/live_metrics.csv fails the
        // first 2 read attempts (a transient error, not a "wrong file"
        // error), then succeeds normally on the 3rd+. A reliable backup
        // with identical content exists at data/archive/metrics_backup.csv,
        // discoverable via LIST_DIRECTORY, but the prompt does not mention
        // it -- whether an agent finds it (or instead just keeps/stops
        // retrying the primary) is exactly what's being tested, under
        // each governance condition.
        tasks.add(new MockTaskSpec(
                "H005_flaky_tool_recovery",
                """
                Read data/live_metrics.csv and report the post-error burden.
                """,
                "data/live_metrics.csv",
                "The post-error burden is 4.2.",
                files(Map.of(
                        "data/live_metrics.csv",
                        """
                        run_id,post_error_burden
                        live001,4.2
                        """,
                        "data/archive/metrics_backup.csv",
                        """
                        run_id,post_error_burden
                        live001,4.2
                        """
                )),
                null,
                Set.of("data/live_metrics.csv", "data/archive/metrics_backup.csv"),
                Map.of("data/live_metrics.csv", 2)
        ));

        return tasks;
    }

    /**
     * Escalation-prone task tier, added for the real-LLM induced-cost pilot.
     * The pilot found that on Gemini only T001/T002 drove the governor into
     * sustained escalation (high regulation intensity), because those are the
     * only tasks whose prompt confidently points at a NONEXISTENT path and the
     * real file sits somewhere non-obvious -- so the model makes several failed
     * reads before it lists and recovers, and each failed read is what pushes
     * the governor NORMAL -> DISTURBED -> RECOVERING. Tasks like T005/T007
     * never engaged, because the model could list-and-find in a single success
     * (disambiguation among existing files, or a one-guess recovery).
     *
     * <p>Each task here is built to the empirically-engaging pattern, dialed up:
     * a confident wrong-path hint PLUS two or three plausible-but-nonexistent
     * candidate paths (so a capable model tends to try several before listing),
     * with the correct file under a non-obvious name/location. The goal is that
     * more than two tasks generate real failed-read escalation on Gemini, so
     * the induced-cost mechanism engages on more than the two anchor tasks.
     *
     * <p><b>Designed, not yet validated on Gemini.</b> Unlike the seed fix
     * (verifiable offline), whether these actually escalate on a real model can
     * only be confirmed by running them -- the mock clings and escalates on
     * everything, so it cannot predict Gemini's behavior. Treat the engagement
     * as intended-by-construction until the pilot data shows it.
     */
    public static List<MockTaskSpec> escalationProneTasks() {
        List<MockTaskSpec> tasks = new ArrayList<>();

        tasks.add(new MockTaskSpec(
                "E001_wrong_root_file",
                """
                Read the calibration file and report the sensor offset value.
                It is probably calibration.json in the project root, or
                possibly config/calibration.json.
                """,
                "config/calibration_v2.json",
                "The sensor offset is 0.42.",
                files(Map.of(
                        "config/calibration_v2.json",
                        """
                        {
                          "sensor_offset": 0.42,
                          "status": "active"
                        }
                        """
                ))
        ));

        tasks.add(new MockTaskSpec(
                "E002_wrong_dir_and_ext",
                """
                Read data/output.json and report the final accuracy.
                The results were exported as JSON to the data folder;
                if not there, try results.json.
                """,
                "analysis/output_final.csv",
                "The final accuracy is 0.87.",
                files(Map.of(
                        "analysis/output_final.csv",
                        """
                        metric,value
                        final_accuracy,0.87
                        """
                ))
        ));

        // Fixed after the k-sweep run: the original "results.csv OR metrics.csv
        // under the reports folder" phrasing let Gemini LIST reports/ and
        // find the file without any failed read (g stayed ~0.31, no
        // engagement). A confident single wrong path lures the failed read
        // that drives escalation, matching the T002/E002 pattern that worked.
        tasks.add(new MockTaskSpec(
                "E003_confident_wrong_report",
                """
                Read reports/results.csv and report the post-error burden.
                The completed run's metrics are at reports/results.csv.
                """,
                "reports/run_report_final.csv",
                "The post-error burden is 2.7.",
                files(Map.of(
                        "reports/run_report_final.csv",
                        """
                        run_id,status,post_error_burden
                        run_final,complete,2.7
                        """
                ))
        ));

        tasks.add(new MockTaskSpec(
                "E004_deep_nest_wrong_hint",
                """
                Read the final run log and report the exit status.
                It should be at logs/final.log, or possibly logs/latest.log.
                """,
                "logs/2026/session_last/run.log",
                "The run exited with status success.",
                files(Map.of(
                        "logs/2026/session_last/run.log",
                        """
                        run_id=run_last
                        exit_status=success
                        post_error_burden=1.9
                        """
                ))
        ));

        // C-series: COLLAPSE-CAPABLE tasks -- built to the empirical recipe of
        // the two tasks (T002, E002) that both escalated AND stayed solvable on
        // Gemini, so the induced cost can knock success DOWN (amplification) and
        // cost-aware backoff can bring it back UP (rescue). The recipe: a
        // confident wrong-path hint (drives the failed read -> escalation) PLUS
        // a SHORT, verbatim-reproducible answer (correctness is substring match
        // of the expected answer, so long/specific answers like T001's are
        // effectively unsolvable regardless of finding the file). Added for the
        // powered rerun so the rescue result rests on more than one task.

        tasks.add(new MockTaskSpec(
                "C001_wrong_subdir",
                """
                Read scores/summary.csv and report the F1 score.
                The scores are probably at scores/summary.csv.
                """,
                "scores/final/summary.csv",
                "The F1 score is 0.91.",
                files(Map.of(
                        "scores/final/summary.csv",
                        """
                        metric,f1_score
                        run_final,0.91
                        """
                ))
        ));

        tasks.add(new MockTaskSpec(
                "C002_wrong_ext_dir",
                """
                Read logs/throughput.json and report the peak throughput.
                The data was saved as JSON in the logs folder; if not there,
                try throughput.json.
                """,
                "telemetry/throughput_final.csv",
                "The peak throughput is 1450.",
                files(Map.of(
                        "telemetry/throughput_final.csv",
                        """
                        metric,value
                        peak_throughput,1450
                        """
                ))
        ));

        tasks.add(new MockTaskSpec(
                "C003_wrong_root",
                """
                Read latency.json and report the median latency.
                It is probably latency.json in the project root.
                """,
                "benchmarks/latency_v2.json",
                "The median latency is 34.",
                files(Map.of(
                        "benchmarks/latency_v2.json",
                        """
                        {
                          "median_latency": 34,
                          "unit": "ms"
                        }
                        """
                ))
        ));

        return tasks;
    }

    private static Map<String, String> files(Map<String, String> input) {
        return new LinkedHashMap<>(input);
    }
}
