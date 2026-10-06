/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package org.cas.mlselect.data;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Generic reconstructed candidate event.
 *
 * This class deliberately avoids channel-specific names such as Omega, Lambda,
 * K+, missing mass, etc. Physics-specific quantities should be stored as named
 * variables in the variables map.
 * 
 * @author veronique
 */
public final class CandidateEvent {
    private final long eventId;
    private final int run;
    private final Map<String, Double> variables;
    private final Map<String, String> tags;
    private final double weight;

    private CandidateEvent(Builder builder) {
        this.eventId = builder.eventId;
        this.run = builder.run;
        this.variables = Collections.unmodifiableMap(new HashMap<>(builder.variables));
        this.tags = Collections.unmodifiableMap(new HashMap<>(builder.tags));
        this.weight = builder.weight;
    }

    public long eventId() { return eventId; }
    public int run() { return run; }
    public double weight() { return weight; }

    public Map<String, Double> variables() { return variables; }
    public Map<String, String> tags() { return tags; }

    public double get(String name) {
        Double value = variables.get(name);
        if (value == null) {
            throw new IllegalArgumentException("Missing event variable: " + name);
        }
        return value;
    }

    public double getOrDefault(String name, double defaultValue) {
        return variables.getOrDefault(name, defaultValue);
    }

    public String tag(String name) {
        return tags.get(name);
    }

    public static Builder builder(long eventId, int run) {
        return new Builder(eventId, run);
    }

    public static final class Builder {
        private final long eventId;
        private final int run;
        private final Map<String, Double> variables = new HashMap<>();
        private final Map<String, String> tags = new HashMap<>();
        private double weight = 1.0;

        private Builder(long eventId, int run) {
            this.eventId = eventId;
            this.run = run;
        }

        public Builder put(String name, double value) {
            variables.put(Objects.requireNonNull(name), value);
            return this;
        }

        public Builder tag(String name, String value) {
            tags.put(Objects.requireNonNull(name), Objects.requireNonNull(value));
            return this;
        }

        public Builder weight(double weight) {
            this.weight = weight;
            return this;
        }

        public CandidateEvent build() {
            return new CandidateEvent(this);
        }
    }
}
