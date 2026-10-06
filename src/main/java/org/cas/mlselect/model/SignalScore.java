/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package org.cas.mlselect.model;

/**
 * ML model output. The score should normally be in [0,1], where larger means
 * more signal-like. Uncertainty is optional but useful for regulation.
 * 
 * @author veronique
 */
public record SignalScore(double score, double uncertainty) {
    public SignalScore {
        if (!Double.isFinite(score)) throw new IllegalArgumentException("score must be finite");
        if (!Double.isFinite(uncertainty)) throw new IllegalArgumentException("uncertainty must be finite");
    }

    public static SignalScore certain(double score) {
        return new SignalScore(score, 0.0);
    }
}
