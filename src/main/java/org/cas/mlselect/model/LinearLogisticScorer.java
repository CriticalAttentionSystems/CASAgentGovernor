/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package org.cas.mlselect.model;

import org.cas.mlselect.features.FeatureVector;

import java.util.Arrays;

/**
 * Tiny built-in scorer for testing and simple baselines.
 * In production this can be replaced by an ONNX, XGBoost, TMVA, or custom scorer.
 * 
 * @author veronique
 */
public final class LinearLogisticScorer implements SignalScorer {
    private final double bias;
    private final double[] weights;

    public LinearLogisticScorer(double bias, double[] weights) {
        this.bias = bias;
        this.weights = weights.clone();
    }

    @Override
    public SignalScore score(FeatureVector features) {
        if (features.size() != weights.length) {
            throw new IllegalArgumentException("Expected " + weights.length + " features, got " + features.size());
        }
        double z = bias;
        for (int i = 0; i < weights.length; i++) {
            z += weights[i] * features.value(i);
        }
        double p = 1.0 / (1.0 + Math.exp(-z));
        double uncertainty = 4.0 * p * (1.0 - p); // 1 near boundary, 0 near confident extremes
        return new SignalScore(p, uncertainty);
    }

    @Override
    public String toString() {
        return "LinearLogisticScorer{" + "bias=" + bias + ", weights=" + Arrays.toString(weights) + '}';
    }
}
