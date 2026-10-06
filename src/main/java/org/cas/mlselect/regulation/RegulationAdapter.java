/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package org.cas.mlselect.regulation;

/**
 * Bridge between the ML selector and a regulation system.
 *
 * Implement this interface inside the IRAM-OMEGA-Q project or create a small
 * adapter module that depends on both packages.
 * 
 * @author veronique
 */
public interface RegulationAdapter {
    RegulationState update(RegulationInput input);
}
