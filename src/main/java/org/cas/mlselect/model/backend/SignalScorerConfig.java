/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package org.cas.mlselect.model.backend;

import java.nio.file.Path;

/** 
 * Configuration for choosing the ML scorer backend without changing selection code. 
 * 
 * @author veronique
 */
public record SignalScorerConfig(
        SignalModelBackend backend,
        String scoreColumnName,
        Path onnxModelPath,
        String onnxInputName,
        String onnxOutputName,
        boolean squashOrSigmoid
) {
    public static SignalScorerConfig tmvaBdtScoreColumn(String scoreColumnName) {
        return new SignalScorerConfig(
                SignalModelBackend.TMVA_BDT_SCORE_COLUMN,
                scoreColumnName,
                null,
                null,
                null,
                false);
    }

    public static SignalScorerConfig xgboostScoreColumn(String scoreColumnName) {
        return new SignalScorerConfig(
                SignalModelBackend.XGBOOST_SCORE_COLUMN,
                scoreColumnName,
                null,
                null,
                null,
                false);
    }

    public static SignalScorerConfig xgboostOnnx(Path modelPath) {
        return new SignalScorerConfig(
                SignalModelBackend.ONNX_RUNTIME,
                null,
                modelPath,
                null,
                null,
                false);
    }

    public SignalScorerConfig withSquashOrSigmoid(boolean enabled) {
        return new SignalScorerConfig(backend, scoreColumnName, onnxModelPath, onnxInputName, onnxOutputName, enabled);
    }

    public SignalScorerConfig withOnnxNames(String inputName, String outputName) {
        return new SignalScorerConfig(backend, scoreColumnName, onnxModelPath, inputName, outputName, squashOrSigmoid);
    }
}
