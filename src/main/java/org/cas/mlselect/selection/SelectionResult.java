/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package org.cas.mlselect.selection;

import org.cas.mlselect.metrics.SelectionWindowMetrics;
import org.cas.mlselect.model.SignalScore;
import org.cas.mlselect.regulation.RegulationState;

/**
 * @author veronique
 */
public record SelectionResult(
        SelectionDecision decision,
        SignalScore score,
        double threshold,
        RegulationState regulationState,
        SelectionWindowMetrics metrics
) {
    public boolean accepted() {
        return decision == SelectionDecision.ACCEPT_SIGNAL_CANDIDATE;
    }
}
