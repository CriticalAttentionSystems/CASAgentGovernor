/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package org.cas.mlselect.example;

import org.cas.mlselect.data.CandidateEvent;
import org.cas.mlselect.features.NamedVariableFeatureExtractor;
import org.cas.mlselect.model.LinearLogisticScorer;
import org.cas.mlselect.regulation.LocalIramStyleRegulationAdapter;
import org.cas.mlselect.selection.RegulatedSignalSelector;
import org.cas.mlselect.selection.SelectionResult;

import java.util.List;

/**
 * 
 * @author veronique
 */
public final class ExampleMain {
    public static void main(String[] args) {
        var features = new NamedVariableFeatureExtractor(List.of(
                "missing_mass_pull", "pid_chi2", "missing_pt", "vertex_dt"
        ));

        var scorer = new LinearLogisticScorer(
                1.5,
                new double[] {-1.2, -0.4, -0.8, -0.3}
        );

        var regulation = new LocalIramStyleRegulationAdapter(
                0.20, 0.95,
                0.10, 0.20, 0.05, 0.10,
                0.0
        );

        var selector = new RegulatedSignalSelector(features, scorer, regulation, 0.65, 50);

        CandidateEvent candidate = CandidateEvent.builder(1001, 16042)
                .put("missing_mass_pull", 0.8)
                .put("pid_chi2", 1.2)
                .put("missing_pt", 0.15)
                .put("vertex_dt", 0.3)
                .tag("sector", "3")
                .build();

        SelectionResult result = selector.select(candidate);
        System.out.printf("score=%.4f threshold=%.4f accepted=%s status=%s%n",
                result.score().score(),
                result.threshold(),
                result.accepted(),
                result.regulationState().status());
    }
}
