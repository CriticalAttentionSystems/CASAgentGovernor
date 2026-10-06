/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package org.cas.mlselect.metrics;

import org.cas.mlselect.model.SignalScore;

import java.util.ArrayDeque;
import java.util.Deque;

/**
 * Simple rolling metric accumulator. More sophisticated versions can compare
 * sidebands, MC expectation, sector/run bins, or kinematic bins.
 * 
 * @author veronique
 */
public final class WindowMetricAccumulator {
    private final int windowSize;
    private final Deque<Double> scores = new ArrayDeque<>();
    private final Deque<Double> uncertainties = new ArrayDeque<>();
    private double lastMeanScore = Double.NaN;

    public WindowMetricAccumulator(int windowSize) {
        if (windowSize < 2) throw new IllegalArgumentException("windowSize must be at least 2");
        this.windowSize = windowSize;
    }

    public void add(SignalScore score) {
        scores.addLast(score.score());
        uncertainties.addLast(score.uncertainty());
        while (scores.size() > windowSize) scores.removeFirst();
        while (uncertainties.size() > windowSize) uncertainties.removeFirst();
    }

    public SelectionWindowMetrics snapshot(double externalMismatch, double residualBurden) {
        if (scores.isEmpty()) return SelectionWindowMetrics.empty();
        double meanScore = mean(scores);
        double meanUncertainty = mean(uncertainties);
        double volatility = stddev(scores, meanScore);
        double momentum = Double.isNaN(lastMeanScore) ? 0.0 : meanScore - lastMeanScore;
        lastMeanScore = meanScore;

        double capacity = clamp01(1.0 - meanUncertainty);
        double load = clamp01(meanUncertainty + volatility);

        return new SelectionWindowMetrics(
                capacity,
                load,
                externalMismatch,
                momentum,
                volatility,
                residualBurden,
                meanScore,
                meanUncertainty,
                scores.size()
        );
    }

    private static double mean(Deque<Double> xs) {
        double sum = 0.0;
        for (double x : xs) sum += x;
        return sum / xs.size();
    }

    private static double stddev(Deque<Double> xs, double mean) {
        double sum = 0.0;
        for (double x : xs) {
            double d = x - mean;
            sum += d * d;
        }
        return Math.sqrt(sum / xs.size());
    }

    private static double clamp01(double x) {
        return Math.max(0.0, Math.min(1.0, x));
    }
}
