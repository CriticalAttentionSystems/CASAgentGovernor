/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package org.cas.mlselect.features;

import org.cas.mlselect.data.CandidateEvent;

import java.util.List;

/**
 * @author veronique
 */
/** Extracts features directly from named variables stored in CandidateEvent. */
public final class NamedVariableFeatureExtractor implements FeatureExtractor {
    private final List<String> featureNames;

    public NamedVariableFeatureExtractor(List<String> featureNames) {
        this.featureNames = List.copyOf(featureNames);
    }

    @Override
    public FeatureVector extract(CandidateEvent event) {
        double[] values = new double[featureNames.size()];
        for (int i = 0; i < featureNames.size(); i++) {
            values[i] = event.get(featureNames.get(i));
        }
        return new FeatureVector(featureNames, values);
    }
}
