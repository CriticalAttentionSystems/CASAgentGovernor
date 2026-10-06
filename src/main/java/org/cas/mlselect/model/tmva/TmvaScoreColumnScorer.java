/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package org.cas.mlselect.model.tmva;

import org.cas.mlselect.features.FeatureVector;
import org.cas.mlselect.model.SignalScore;
import org.cas.mlselect.model.SignalScorer;

import java.util.Objects;

/**
 * Lightweight scorer for the common production workflow where ROOT/TMVA has
 * already written a score into the event ntuple or CSV file.
 *
 * Use this when you score a whole TTree with a ROOT macro and then read the
 * resulting score branch back into the Java regulated selector. It is much
 * faster and safer than launching ROOT once per event.
 * 
 * @author veronique
 */
public final class TmvaScoreColumnScorer implements SignalScorer {
    private final String scoreFeatureName;
    private final boolean squashToUnitInterval;

    public TmvaScoreColumnScorer(String scoreFeatureName) {
        this(scoreFeatureName, false);
    }

    public TmvaScoreColumnScorer(String scoreFeatureName, boolean squashToUnitInterval) {
        this.scoreFeatureName = Objects.requireNonNull(scoreFeatureName);
        this.squashToUnitInterval = squashToUnitInterval;
    }

    @Override
    public SignalScore score(FeatureVector features) {
        int index = features.names().indexOf(scoreFeatureName);
        if (index < 0) {
            throw new IllegalArgumentException("FeatureVector does not contain TMVA score feature: " + scoreFeatureName);
        }
        double raw = features.value(index);
        double score = squashToUnitInterval ? squash(raw) : raw;
        double uncertainty = 4.0 * score * (1.0 - score);
        return new SignalScore(score, uncertainty);
    }

    private static double squash(double raw) {
        if (raw >= 0.0 && raw <= 1.0) return raw;
        return 1.0 / (1.0 + Math.exp(-raw));
    }
}
