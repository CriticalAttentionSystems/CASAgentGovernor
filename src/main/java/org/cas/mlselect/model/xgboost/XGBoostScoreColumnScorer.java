/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package org.cas.mlselect.model.xgboost;

import org.cas.mlselect.features.FeatureVector;
import org.cas.mlselect.model.SignalScore;
import org.cas.mlselect.model.SignalScorer;

import java.util.Objects;

/**
 * Production scorer for the workflow where an XGBoost model has already scored
 * the ntuple and written a branch/column such as {@code xgb_score}.
 *
 * This is useful when training and batch scoring are done in Python, while Java
 * only performs regulated selection and IRAM-OMEGA-Q interfacing.
 * 
 * @author veronique
 */
public final class XGBoostScoreColumnScorer implements SignalScorer {
    private final String scoreFeatureName;
    private final boolean squashToUnitInterval;

    public XGBoostScoreColumnScorer(String scoreFeatureName) {
        this(scoreFeatureName, false);
    }

    public XGBoostScoreColumnScorer(String scoreFeatureName, boolean squashToUnitInterval) {
        this.scoreFeatureName = Objects.requireNonNull(scoreFeatureName, "scoreFeatureName");
        this.squashToUnitInterval = squashToUnitInterval;
    }

    @Override
    public SignalScore score(FeatureVector features) {
        int index = features.names().indexOf(scoreFeatureName);
        if (index < 0) {
            throw new IllegalArgumentException("FeatureVector does not contain XGBoost score feature: " + scoreFeatureName);
        }
        double raw = features.value(index);
        double score = squashToUnitInterval ? squash(raw) : clamp01(raw);
        double uncertainty = 4.0 * score * (1.0 - score);
        return new SignalScore(score, uncertainty);
    }

    private static double squash(double raw) {
        if (raw >= 0.0 && raw <= 1.0) return raw;
        return 1.0 / (1.0 + Math.exp(-raw));
    }

    private static double clamp01(double x) {
        if (x < 0.0) return 0.0;
        if (x > 1.0) return 1.0;
        return x;
    }
}
