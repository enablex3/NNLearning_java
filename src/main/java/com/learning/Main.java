package com.learning;

import java.util.concurrent.ThreadLocalRandom;

import org.encog.engine.network.activation.ActivationSigmoid;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.ml.train.MLTrain;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;
import org.encog.util.arrayutil.NormalizationAction;
import org.encog.util.arrayutil.NormalizeArray;
import org.encog.util.arrayutil.NormalizedField;

public class Main {

    public static void main(String[] args) {
        //runXORExample();
        //runLunarNormalExample();
        runMemArrayNormalExample();
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
        System.out.println("XOR Neural Network results:");
        for (MLDataPair pair : trainingSet) {
            // Get output of next pair
            final MLData outData = network.compute(pair.getInput());
            System.out.println(pair.getInput().getData(0) + 
                "," + 
                pair.getInput().getData(1) + 
                ", actual=" + 
                outData.getData(0) + 
                ",ideal=" + 
                pair.getIdeal().getData(0)
            );
        }
    }

    private static void runLunarNormalExample() {
        // Create normalized field for fuel stats
        // Acceptable range (0-200)
        // Normalization range (-0.9-0.9)
        System.out.println("Fuel Stat Normalization");
        NormalizedField fuelStats = new NormalizedField(
            NormalizationAction.Normalize,
            "fuel",
            200,
            0,
            -0.9,
            0.9
        );
        // Normalize some values
        double n100 = fuelStats.normalize(100);
        double n125 = fuelStats.normalize(125);
        double n3 = fuelStats.normalize(3);
        System.out.println("100 -> " + n100 + ", 125 -> " + n125 + ", 3 -> " + n3);
    }

    private static void runMemArrayNormalExample() {
        // Generate array of random values
        double[] randDoubles = new double[20];
        for (int i = 0; i < randDoubles.length; i++) {
            double nextDouble = ThreadLocalRandom.current().nextDouble(0, 119);
            randDoubles[i] = nextDouble;
        }
        // Create normal array
        NormalizeArray normalizeArray = new NormalizeArray();
        normalizeArray.setNormalizedHigh(0.9);
        normalizeArray.setNormalizedLow(-0.9);
        // Normalize random data
        double[] normalizedRandomData = normalizeArray.process(randDoubles);
        for (int i = 0; i < normalizedRandomData.length; i++) {
            System.out.println(randDoubles[i] + " -> " + normalizedRandomData[i]);
        }
    }
}
