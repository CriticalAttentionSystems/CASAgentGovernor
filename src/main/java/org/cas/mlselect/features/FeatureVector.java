/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package org.cas.mlselect.features;

import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * 
 * @author veronique
 */
public final class FeatureVector {
    private final List<String> names;
    private final double[] values;

    public FeatureVector(List<String> names, double[] values) {
        this.names = List.copyOf(Objects.requireNonNull(names));
        this.values = Objects.requireNonNull(values).clone();
        if (this.names.size() != this.values.length) {
            throw new IllegalArgumentException("Feature name count does not match feature value count.");
        }
    }

    public List<String> names() { return names; }
    public double[] values() { return values.clone(); }
    public int size() { return values.length; }

    public double value(int index) { return values[index]; }

    @Override
    public String toString() {
        return "FeatureVector{" + "names=" + names + ", values=" + Arrays.toString(values) + '}';
    }
}
