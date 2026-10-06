/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package org.cas.mlselect.regulation;

import org.cas.mlselect.metrics.SelectionWindowMetrics;

/**
 * Local IRAM-style controller for development when the full IRAM-OMEGA-Q
 * simulator is not linked. It adjusts threshold according to uncertainty,
 * mismatch, volatility, and residual burden.
 * 
 * @author veronique
 */
public final class LocalIramStyleRegulationAdapter implements RegulationAdapter {
    private final double minThreshold;
    private final double maxThreshold;
    private final double alphaLoad;
    private final double betaMismatch;
    private final double gammaRecovery;
    private final double deltaBurden;
    private double gain;

    public LocalIramStyleRegulationAdapter(
            double minThreshold,
            double maxThreshold,
            double alphaLoad,
            double betaMismatch,
            double gammaRecovery,
            double deltaBurden,
            double initialGain
    ) {
        this.minThreshold = minThreshold;
        this.maxThreshold = maxThreshold;
        this.alphaLoad = alphaLoad;
        this.betaMismatch = betaMismatch;
        this.gammaRecovery = gammaRecovery;
        this.deltaBurden = deltaBurden;
        this.gain = initialGain;
    }

    @Override
    public RegulationState update(RegulationInput input) {
        SelectionWindowMetrics m = input.metrics();

        double pressure =
                alphaLoad * m.load()
              + betaMismatch * m.mismatch()
              + deltaBurden * m.residualBurden()
              + 0.5 * m.volatility()
              - gammaRecovery * m.capacity();

        gain = clamp(gain + 0.05 * pressure, -1.0, 1.0);
        double next = clamp(input.baseThreshold() + gain, minThreshold, maxThreshold);

        String status = pressure > 0.05 ? "TIGHTENING"
                : pressure < -0.05 ? "RELAXING"
                : "STABLE";

        return new RegulationState(next, gain, pressure, status);
    }

    private static double clamp(double x, double lo, double hi) {
        return Math.max(lo, Math.min(hi, x));
    }
}
