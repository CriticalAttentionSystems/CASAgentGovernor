/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package org.cas.mlselect.io;

import org.cas.mlselect.data.CandidateEvent;

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * @author veronique
 */
/** Minimal CSV reader for quick tests. Header must include run,event. */
public final class CsvCandidateReader {
    public List<CandidateEvent> read(Path path) throws IOException {
        try (BufferedReader br = Files.newBufferedReader(path)) {
            String header = br.readLine();
            if (header == null) return List.of();
            String[] names = header.split(",");
            int runIndex = indexOf(names, "run");
            int eventIndex = indexOf(names, "event");
            if (runIndex < 0 || eventIndex < 0) {
                throw new IllegalArgumentException("CSV must contain run,event columns.");
            }

            List<CandidateEvent> events = new ArrayList<>();
            String line;
            while ((line = br.readLine()) != null) {
                if (line.isBlank()) continue;
                String[] values = line.split(",");
                int run = Integer.parseInt(values[runIndex].trim());
                long event = Long.parseLong(values[eventIndex].trim());
                CandidateEvent.Builder b = CandidateEvent.builder(event, run);
                for (int i = 0; i < Math.min(names.length, values.length); i++) {
                    if (i == runIndex || i == eventIndex) continue;
                    try {
                        b.put(names[i].trim(), Double.parseDouble(values[i].trim()));
                    } catch (NumberFormatException ignored) {
                        b.tag(names[i].trim(), values[i].trim());
                    }
                }
                events.add(b.build());
            }
            return events;
        }
    }

    private static int indexOf(String[] names, String target) {
        for (int i = 0; i < names.length; i++) {
            if (names[i].trim().equals(target)) return i;
        }
        return -1;
    }
}
