package com.learning;

import org.encog.engine.network.activation.ActivationSigmoid;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.ml.train.MLTrain;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;

public class Main {

    public static void main(String[] args) {
        runXORExample();
    }

    private static void runXORExample() {
        // First example (page 12) - XOR NN
        BasicNetwork network = new BasicNetwork();
        network.addLayer(new BasicLayer(null, true, 2));
        network.addLayer(new BasicLayer(new ActivationSigmoid(), true, 3));
        network.addLayer(new BasicLayer(new ActivationSigmoid(), false, 1));
        // No more layers can be added
        network.getStructure().finalizeStructure();
        // Randomizes weights in connections between layers
        network.reset();
        // Create data for this network
        double[][] xorInput = { {0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}, {1.0, 1.0} };
        double[][] xorOutput = { {0.0}, {1.0}, {1.0}, {0.0} };
        // Create training set
        MLDataSet trainingSet = new BasicMLDataSet(xorInput, xorOutput);
        // Create trainer
        MLTrain trainer = new ResilientPropagation(network, trainingSet);
        // Train the network
        int epoch = 1;
        double errPercentile = 0.01;
        do {
            trainer.iteration();
            System.out.println("Epoch #" + epoch + " Error: " + trainer.getError());
            epoch++;
        } while (trainer.getError() > errPercentile);
        // Network is ready for use once it's been trained
    }
}
