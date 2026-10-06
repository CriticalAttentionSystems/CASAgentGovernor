/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package org.cas.mlselect.metrics;

/**
 * Rolling or batch-level metrics passed into the regulation layer.
 * These names are intentionally compatible with BioReg/IRAM-style monitoring.
 * 
 * @author veronique
 */
public record SelectionWindowMetrics(
        double capacity,
        double load,
        double mismatch,
        double momentum,
        double volatility,
        double residualBurden,
        double meanScore,
        double meanUncertainty,
        int eventCount
) {
    public static SelectionWindowMetrics empty() {
        return new SelectionWindowMetrics(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0);
    }
}
