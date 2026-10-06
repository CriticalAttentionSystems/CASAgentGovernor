/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package org.cas.agentgovernor.tools;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.InvalidPathException;
import java.nio.file.Path;
import java.util.Comparator;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import org.cas.agentgovernor.core.ToolAction;
import org.cas.agentgovernor.core.ToolResult;

/**
 *
 * @author veronique
 */

public class FileTool implements AgentTool {

    private final Path root;

    public FileTool(String rootDirectory) {
        if (rootDirectory == null || rootDirectory.isBlank()) {
            throw new IllegalArgumentException("Root directory cannot be blank.");
        }

        this.root = Path.of(rootDirectory).toAbsolutePath().normalize();
    }

    @Override
    public String name() {
        return "file";
    }

    @Override
    public ToolResult execute(ToolAction action) {
        String actionType = action.actionType();

        if ("READ_FILE".equalsIgnoreCase(actionType)) {
            return readFile(action.target());
        }

        if ("LIST_DIRECTORY".equalsIgnoreCase(actionType)) {
            return listDirectory(action.target());
        }

        return ToolResult.failure(
                "UNSUPPORTED_FILE_ACTION",
                "Unsupported file action: " + actionType
        );
    }

    private ToolResult readFile(String target) {
        Path path = resolveInsideRoot(target);

        if (path == null) {
            return ToolResult.failure("INVALID_PATH", "Invalid or unsafe file path: " + target);
        }

        if (!Files.exists(path)) {
            return ToolResult.failure("FILE_NOT_FOUND", "File not found: " + target);
        }

        if (Files.isDirectory(path)) {
            return ToolResult.failure("IS_DIRECTORY", "Path is a directory, not a file: " + target);
        }

        try {
            String text = Files.readString(path, StandardCharsets.UTF_8);
            return ToolResult.success(text);
        } catch (IOException e) {
            return ToolResult.failure("READ_ERROR", e.getMessage());
        }
    }

    private ToolResult listDirectory(String target) {
        String normalizedTarget = target == null || target.isBlank() ? "." : target;
        Path path = resolveInsideRoot(normalizedTarget);

        if (path == null) {
            return ToolResult.failure("INVALID_PATH", "Invalid or unsafe directory path: " + target);
        }

        if (!Files.exists(path)) {
            return ToolResult.failure("DIRECTORY_NOT_FOUND", "Directory not found: " + target);
        }

        if (!Files.isDirectory(path)) {
            return ToolResult.failure("NOT_DIRECTORY", "Path is not a directory: " + target);
        }

        try (Stream<Path> stream = Files.list(path)) {
            String listing = stream
                    .sorted(Comparator.comparing(p -> p.getFileName().toString()))
                    .map(p -> {
                        String name = p.getFileName().toString();
                        return Files.isDirectory(p) ? name + "/" : name;
                    })
                    .collect(Collectors.joining("\n"));

            if (listing.isBlank()) {
                listing = "(empty directory)";
            }

            return ToolResult.success(listing);

        } catch (IOException e) {
            return ToolResult.failure("LIST_ERROR", e.getMessage());
        }
    }

    private Path resolveInsideRoot(String target) {
        try {
            String safeTarget = target == null || target.isBlank() ? "." : target;
            Path resolved = root.resolve(safeTarget).normalize();

            if (!resolved.startsWith(root)) {
                return null;
            }

            return resolved;
        } catch (InvalidPathException e) {
            return null;
        }
    }
}