/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package org.cas.mlselect.model.backend;

/** 
 * Available first-line model backends for regulated event selection. 
 * 
 * @author veronique
 */
public enum SignalModelBackend {
    /** ROOT/TMVA BDT score already stored as a branch or CSV column. */
    TMVA_BDT_SCORE_COLUMN,

    /** XGBoost score already stored as a branch or CSV column. */
    XGBOOST_SCORE_COLUMN,

    /** XGBoost, TensorFlow, or other model exported to ONNX and evaluated in Java. */
    ONNX_RUNTIME
}
