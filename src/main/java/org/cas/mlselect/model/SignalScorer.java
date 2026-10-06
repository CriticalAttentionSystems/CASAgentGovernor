/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package org.cas.mlselect.model;

import org.cas.mlselect.features.FeatureVector;

@FunctionalInterface
public interface SignalScorer {
    SignalScore score(FeatureVector features);
}
