/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package org.cas.mlselect.features;

import org.cas.mlselect.data.CandidateEvent;

/**
 * 
 * @author veronique
 */
@FunctionalInterface
public interface FeatureExtractor {
    FeatureVector extract(CandidateEvent event);
}
