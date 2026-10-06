/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package org.cas.mlselect.regulation;
/**
 * 
 * @author veronique
 */
public final class NoOpRegulationAdapter implements RegulationAdapter {
    @Override
    public RegulationState update(RegulationInput input) {
        return new RegulationState(input.currentThreshold(), 0.0, 0.0, "NO_REGULATION");
    }
}
