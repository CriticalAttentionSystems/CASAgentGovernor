/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package org.cas.mlselect.model.tmva;

import org.cas.mlselect.features.FeatureVector;
import org.cas.mlselect.model.SignalScore;
import org.cas.mlselect.model.SignalScorer;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Objects;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;

/**
 * SignalScorer implementation that calls a ROOT/TMVA macro in a separate process.
 *
 * This is intentionally a bridge, not a Java reimplementation of TMVA. TMVA is
 * ROOT-native C++; this class lets the generic Java selector use a trained TMVA
 * XML weights file while keeping the Java package independent of ROOT libraries.
 *
 * For production over millions of candidates, prefer batch scoring: write a ROOT
 * macro that scores a TTree/CSV once and then load the score branch into Java.
 * This class is best for integration tests, low-rate online checks, and proving
 * the IRAM-OMEGA-Q regulation interface.
 * 
 * @author veronique
 */
public final class TmvaExternalProcessScorer implements SignalScorer {
    private final Path rootMacro;
    private final Path weightsXml;
    private final String methodName;
    private final String rootExecutable;
    private final Duration timeout;
    private final boolean squashToUnitInterval;

    public TmvaExternalProcessScorer(Path rootMacro, Path weightsXml, String methodName) {
        this(rootMacro, weightsXml, methodName, "root", Duration.ofSeconds(10), true);
    }

    public TmvaExternalProcessScorer(
            Path rootMacro,
            Path weightsXml,
            String methodName,
            String rootExecutable,
            Duration timeout,
            boolean squashToUnitInterval
    ) {
        this.rootMacro = Objects.requireNonNull(rootMacro);
        this.weightsXml = Objects.requireNonNull(weightsXml);
        this.methodName = Objects.requireNonNull(methodName);
        this.rootExecutable = Objects.requireNonNull(rootExecutable);
        this.timeout = Objects.requireNonNull(timeout);
        this.squashToUnitInterval = squashToUnitInterval;
    }

    @Override
    public SignalScore score(FeatureVector features) {
        String names = String.join(",", features.names());
        String values = joinValues(features.values());
        String macroCall = String.format(Locale.ROOT,
                "%s(\"%s\",\"%s\",\"%s\",\"%s\")",
                rootMacro.toAbsolutePath(),
                weightsXml.toAbsolutePath(),
                escape(methodName),
                escape(names),
                escape(values));

        List<String> command = new ArrayList<>();
        command.add(rootExecutable);
        command.add("-l");
        command.add("-b");
        command.add("-q");
        command.add(macroCall);

        ProcessBuilder pb = new ProcessBuilder(command);
        pb.redirectErrorStream(true);

        try {
            Process process = pb.start();
            boolean finished = process.waitFor(timeout.toMillis(), TimeUnit.MILLISECONDS);
            String output;
            try (BufferedReader br = new BufferedReader(new InputStreamReader(process.getInputStream(), StandardCharsets.UTF_8))) {
                output = br.lines().collect(Collectors.joining(System.lineSeparator()));
            }
            if (!finished) {
                process.destroyForcibly();
                throw new IllegalStateException("TMVA ROOT process timed out after " + timeout + ". Output: " + output);
            }
            if (process.exitValue() != 0) {
                throw new IllegalStateException("TMVA ROOT process failed with exit code " + process.exitValue() + ". Output: " + output);
            }
            double raw = parseScore(output);
            double score = squashToUnitInterval ? squash(raw) : raw;
            double uncertainty = 4.0 * score * (1.0 - score);
            return new SignalScore(score, uncertainty);
        } catch (IOException e) {
            throw new IllegalStateException("Could not start ROOT/TMVA process. Is ROOT on PATH?", e);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("Interrupted while waiting for ROOT/TMVA process.", e);
        }
    }

    private static String joinValues(double[] values) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < values.length; i++) {
            if (i > 0) sb.append(',');
            sb.append(Double.toString(values[i]));
        }
        return sb.toString();
    }

    private static String escape(String s) {
        return s.replace("\\", "\\\\").replace("\"", "\\\"");
    }

    private static double parseScore(String output) {
        for (String line : output.split("\\R")) {
            line = line.trim();
            if (line.startsWith("TMVA_SCORE=")) {
                return Double.parseDouble(line.substring("TMVA_SCORE=".length()).trim());
            }
        }
        throw new IllegalStateException("ROOT/TMVA output did not contain TMVA_SCORE=. Output: " + output);
    }

    private static double squash(double raw) {
        if (raw >= 0.0 && raw <= 1.0) return raw;
        return 1.0 / (1.0 + Math.exp(-raw));
    }
}
