/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package org.cas.mlselect.model.onnx;

import org.cas.mlselect.features.FeatureVector;
import org.cas.mlselect.model.SignalScore;
import org.cas.mlselect.model.SignalScorer;

import java.lang.reflect.Array;
import java.lang.reflect.Constructor;
import java.lang.reflect.Method;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;

/**
 * ONNX Runtime scorer for portable Java inference.
 *
 * This class deliberately uses reflection so the core package still compiles
 * without the ONNX Runtime jar. To use it at runtime, add the ONNX Runtime Java
 * dependency to your application classpath, for example {@code ai.onnxruntime:onnxruntime}.
 *
 * The main intended use is:
 *
 * <pre>
 * XGBoost training in Python -> export model.onnx -> Java OnnxSignalScorer -> regulated selector
 * </pre>
 * 
 * @author veronique
 */
public final class OnnxSignalScorer implements SignalScorer, AutoCloseable {
    private final Object environment;
    private final Object session;
    private final String inputName;
    private final String outputName;
    private final boolean applySigmoid;

    public OnnxSignalScorer(Path modelPath) {
        this(modelPath, null, null, false);
    }

    public OnnxSignalScorer(Path modelPath, String inputName, String outputName, boolean applySigmoid) {
        Objects.requireNonNull(modelPath, "modelPath");
        this.applySigmoid = applySigmoid;
        try {
            Class<?> envClass = Class.forName("ai.onnxruntime.OrtEnvironment");
            Class<?> optionsClass = Class.forName("ai.onnxruntime.OrtSession$SessionOptions");
            this.environment = envClass.getMethod("getEnvironment").invoke(null);
            Constructor<?> optionsCtor = optionsClass.getConstructor();
            Object options = optionsCtor.newInstance();
            this.session = environment.getClass()
                    .getMethod("createSession", String.class, optionsClass)
                    .invoke(environment, modelPath.toString(), options);
            this.inputName = inputName != null ? inputName : firstNameFromSet(session, "getInputNames");
            this.outputName = outputName != null ? outputName : firstNameFromSet(session, "getOutputNames");
        } catch (ReflectiveOperationException ex) {
            throw new IllegalStateException(
                    "ONNX Runtime Java is required for OnnxSignalScorer. " +
                    "Add ai.onnxruntime:onnxruntime to the runtime classpath.", ex);
        }
    }

    @Override
    public SignalScore score(FeatureVector features) {
        Object tensor = null;
        Object result = null;
        try {
            float[][] input = new float[1][features.size()];
            double[] values = features.values();
            for (int i = 0; i < values.length; i++) {
                input[0][i] = (float) values[i];
            }

            Class<?> tensorClass = Class.forName("ai.onnxruntime.OnnxTensor");
            Class<?> envClass = Class.forName("ai.onnxruntime.OrtEnvironment");
            Method createTensor = tensorClass.getMethod("createTensor", envClass, Object.class);
            tensor = createTensor.invoke(null, environment, input);

            Map<String, Object> inputs = new HashMap<>();
            inputs.put(inputName, tensor);
            result = session.getClass().getMethod("run", Map.class).invoke(session, inputs);

            Object outputValue = extractOutputValue(result, outputName);
            double raw = firstNumber(outputValue);
            double score = applySigmoid ? sigmoid(raw) : clamp01(raw);
            double uncertainty = 4.0 * score * (1.0 - score);
            return new SignalScore(score, uncertainty);
        } catch (ReflectiveOperationException ex) {
            throw new IllegalStateException("Could not evaluate ONNX model.", ex);
        } finally {
            closeQuietly(result);
            closeQuietly(tensor);
        }
    }

    @Override
    public void close() {
        closeQuietly(session);
    }

    public String inputName() {
        return inputName;
    }

    public String outputName() {
        return outputName;
    }

    private static String firstNameFromSet(Object target, String methodName) throws ReflectiveOperationException {
        @SuppressWarnings("unchecked")
        Set<String> names = (Set<String>) target.getClass().getMethod(methodName).invoke(target);
        if (names.isEmpty()) {
            throw new IllegalStateException("ONNX model has no names from " + methodName);
        }
        return names.iterator().next();
    }

    private static Object extractOutputValue(Object result, String outputName) throws ReflectiveOperationException {
        Object valueContainer;
        try {
            valueContainer = result.getClass().getMethod("get", String.class).invoke(result, outputName);
        } catch (NoSuchMethodException ex) {
            valueContainer = result.getClass().getMethod("get", int.class).invoke(result, 0);
        }
        if (valueContainer instanceof Optional<?> optional) {
            valueContainer = optional.orElseThrow(() -> new IllegalStateException("ONNX output was empty."));
        }
        Method getValue = valueContainer.getClass().getMethod("getValue");
        return getValue.invoke(valueContainer);
    }

    private static double firstNumber(Object value) {
        if (value instanceof Number number) {
            return number.doubleValue();
        }
        Class<?> cls = value.getClass();
        if (!cls.isArray() || Array.getLength(value) == 0) {
            throw new IllegalArgumentException("ONNX output does not contain a numeric value: " + cls.getName());
        }
        return firstNumber(Array.get(value, 0));
    }

    private static double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    private static double clamp01(double x) {
        if (x < 0.0) return 0.0;
        if (x > 1.0) return 1.0;
        return x;
    }

    private static void closeQuietly(Object object) {
        if (object instanceof AutoCloseable closeable) {
            try {
                closeable.close();
            } catch (Exception ignored) {
                // Deliberately ignored during cleanup.
            }
        }
    }
}
