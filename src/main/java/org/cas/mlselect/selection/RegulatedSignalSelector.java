/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package org.cas.mlselect.selection;

import org.cas.mlselect.data.CandidateEvent;
import org.cas.mlselect.features.FeatureExtractor;
import org.cas.mlselect.features.FeatureVector;
import org.cas.mlselect.metrics.SelectionWindowMetrics;
import org.cas.mlselect.metrics.WindowMetricAccumulator;
import org.cas.mlselect.model.SignalScore;
import org.cas.mlselect.model.SignalScorer;
import org.cas.mlselect.regulation.RegulationAdapter;
import org.cas.mlselect.regulation.RegulationInput;
import org.cas.mlselect.regulation.RegulationState;

import java.util.Objects;

/**
 * Main package entry point.
 *
 * The selector is generic: it does not know the physics channel, reaction,
 * detector, or training method. It only knows how to extract features, score an
 * event, query regulation, and make a threshold decision.
 * 
 * @author veronique
 */
public final class RegulatedSignalSelector {
    private final FeatureExtractor featureExtractor;
    private final SignalScorer scorer;
    private final RegulationAdapter regulationAdapter;
    private final WindowMetricAccumulator metrics;
    private final double baseThreshold;
    private double currentThreshold;
    private double previousThreshold;
    private double externalMismatch;
    private double residualBurden;

    public RegulatedSignalSelector(
            FeatureExtractor featureExtractor,
            SignalScorer scorer,
            RegulationAdapter regulationAdapter,
            double baseThreshold,
            int metricWindowSize
    ) {
        this.featureExtractor = Objects.requireNonNull(featureExtractor);
        this.scorer = Objects.requireNonNull(scorer);
        this.regulationAdapter = Objects.requireNonNull(regulationAdapter);
        this.baseThreshold = baseThreshold;
        this.currentThreshold = baseThreshold;
        this.previousThreshold = baseThreshold;
        this.metrics = new WindowMetricAccumulator(metricWindowSize);
    }

    /** Optional external mismatch from sidebands, data/MC comparison, or run monitors. */
    public void setExternalMismatch(double externalMismatch) {
        this.externalMismatch = externalMismatch;
    }

    /** Optional burden penalty, for example from aggressive threshold motion or low acceptance. */
    public void setResidualBurden(double residualBurden) {
        this.residualBurden = residualBurden;
    }

    public SelectionResult select(CandidateEvent event) {
        FeatureVector features = featureExtractor.extract(event);
        SignalScore score = scorer.score(features);
        metrics.add(score);

        SelectionWindowMetrics snapshot = metrics.snapshot(externalMismatch, residualBurden);
        RegulationInput input = new RegulationInput(currentThreshold, baseThreshold, previousThreshold, snapshot);
        RegulationState state = regulationAdapter.update(input);

        previousThreshold = currentThreshold;
        currentThreshold = state.threshold();

        SelectionDecision decision = score.score() >= currentThreshold
                ? SelectionDecision.ACCEPT_SIGNAL_CANDIDATE
                : SelectionDecision.REJECT_BACKGROUND_LIKE;

        return new SelectionResult(decision, score, currentThreshold, state, snapshot);
    }

    public double currentThreshold() {
        return currentThreshold;
    }
}
