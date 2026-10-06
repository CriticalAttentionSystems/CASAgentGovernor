/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package org.cas.mlselect.model.backend;

import org.cas.mlselect.model.SignalScorer;
import org.cas.mlselect.model.onnx.OnnxSignalScorer;
import org.cas.mlselect.model.tmva.TmvaScoreColumnScorer;
import org.cas.mlselect.model.xgboost.XGBoostScoreColumnScorer;

import java.util.Objects;

/** 
 * Factory that keeps TMVA BDT and XGBoost choices interchangeable. 
 *
 * @author veronique
 */
public final class SignalScorerFactory {
    private SignalScorerFactory() {}

    public static SignalScorer create(SignalScorerConfig config) {
        Objects.requireNonNull(config, "config");
        return switch (config.backend()) {
            case TMVA_BDT_SCORE_COLUMN -> new TmvaScoreColumnScorer(
                    require(config.scoreColumnName(), "scoreColumnName"),
                    config.squashOrSigmoid());
            case XGBOOST_SCORE_COLUMN -> new XGBoostScoreColumnScorer(
                    require(config.scoreColumnName(), "scoreColumnName"),
                    config.squashOrSigmoid());
            case ONNX_RUNTIME -> new OnnxSignalScorer(
                    Objects.requireNonNull(config.onnxModelPath(), "onnxModelPath"),
                    config.onnxInputName(),
                    config.onnxOutputName(),
                    config.squashOrSigmoid());
        };
    }

    private static String require(String value, String name) {
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException(name + " is required for this scorer backend.");
        }
        return value;
    }
}
