/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package org.cas.mlselect.regulation;

import org.cas.mlselect.metrics.SelectionWindowMetrics;

/** 
 * Input sent to IRAM-OMEGA-Q or another regulation controller. 
 * 
 * @author veronique
 */
public record RegulationInput(
        double currentThreshold,
        double baseThreshold,
        double previousThreshold,
        SelectionWindowMetrics metrics
) {}
