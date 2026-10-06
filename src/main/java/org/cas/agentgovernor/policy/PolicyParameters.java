/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package org.cas.agentgovernor.policy;

/**
 *
 * @author veronique
 */

public class PolicyParameters {

    private final double retryFailedWeight;
    private final double inspectRootWeight;
    private final double inspectLocalWeight;
    private final double inspectCandidateDirectoryWeight;
    private final double readCandidateWeight;
    private final double randomGuessWeight;
    private final double prematureAnswerWeight;

    public PolicyParameters(
            double retryFailedWeight,
            double inspectRootWeight,
            double inspectLocalWeight,
            double inspectCandidateDirectoryWeight,
            double readCandidateWeight,
            double randomGuessWeight,
            double prematureAnswerWeight
    ) {
        this.retryFailedWeight = retryFailedWeight;
        this.inspectRootWeight = inspectRootWeight;
        this.inspectLocalWeight = inspectLocalWeight;
        this.inspectCandidateDirectoryWeight = inspectCandidateDirectoryWeight;
        this.readCandidateWeight = readCandidateWeight;
        this.randomGuessWeight = randomGuessWeight;
        this.prematureAnswerWeight = prematureAnswerWeight;
    }

    public static PolicyParameters exploratory() {
        return new PolicyParameters(
                0.40,
                2.00,
                1.50,
                2.00,
                2.50,
                0.30,
                0.10
        );
    }

    public static PolicyParameters clinging() {
        return new PolicyParameters(
                2.50,
                1.00,
                1.00,
                1.50,
                2.00,
                0.40,
                0.10
        );
    }

    public static PolicyParameters persistent() {
        return new PolicyParameters(
                5.00,
                0.80,
                0.70,
                1.20,
                1.80,
                0.50,
                0.10
        );
    }

    public static PolicyParameters impulsive() {
        return new PolicyParameters(
                1.50,
                0.90,
                0.80,
                1.00,
                1.80,
                0.50,
                1.30
        );
    }

    public static PolicyParameters noisy() {
        return new PolicyParameters(
                1.80,
                1.00,
                0.90,
                1.20,
                1.80,
                2.00,
                0.30
        );
    }

    public PolicyParameters withRetryMultiplier(double factor) {
        return new PolicyParameters(
                retryFailedWeight * factor,
                inspectRootWeight,
                inspectLocalWeight,
                inspectCandidateDirectoryWeight,
                readCandidateWeight,
                randomGuessWeight,
                prematureAnswerWeight
        );
    }

    public PolicyParameters withInspectMultiplier(double factor) {
        return new PolicyParameters(
                retryFailedWeight,
                inspectRootWeight * factor,
                inspectLocalWeight * factor,
                inspectCandidateDirectoryWeight * factor,
                readCandidateWeight,
                randomGuessWeight,
                prematureAnswerWeight
        );
    }

    public PolicyParameters withReadCandidateMultiplier(double factor) {
        return new PolicyParameters(
                retryFailedWeight,
                inspectRootWeight,
                inspectLocalWeight,
                inspectCandidateDirectoryWeight,
                readCandidateWeight * factor,
                randomGuessWeight,
                prematureAnswerWeight
        );
    }

    public PolicyParameters withPrematureAnswerMultiplier(double factor) {
        return new PolicyParameters(
                retryFailedWeight,
                inspectRootWeight,
                inspectLocalWeight,
                inspectCandidateDirectoryWeight,
                readCandidateWeight,
                randomGuessWeight,
                prematureAnswerWeight * factor
        );
    }

    public double retryFailedWeight() {
        return retryFailedWeight;
    }

    public double inspectRootWeight() {
        return inspectRootWeight;
    }

    public double inspectLocalWeight() {
        return inspectLocalWeight;
    }

    public double inspectCandidateDirectoryWeight() {
        return inspectCandidateDirectoryWeight;
    }

    public double readCandidateWeight() {
        return readCandidateWeight;
    }

    public double randomGuessWeight() {
        return randomGuessWeight;
    }

    public double prematureAnswerWeight() {
        return prematureAnswerWeight;
    }
}