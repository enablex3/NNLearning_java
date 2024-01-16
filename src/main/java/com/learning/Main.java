package com.learning;

import java.io.File;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;

import org.encog.app.analyst.AnalystFileFormat;
import org.encog.app.analyst.EncogAnalyst;
import org.encog.app.analyst.csv.normalize.AnalystNormalizeCSV;
import org.encog.app.analyst.wizard.AnalystWizard;
import org.encog.engine.network.activation.ActivationSigmoid;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.propagation.Propagation;
import org.encog.neural.networks.training.propagation.back.Backpropagation;
import org.encog.neural.networks.training.propagation.manhattan.ManhattanPropagation;
import org.encog.neural.networks.training.propagation.quick.QuickPropagation;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;
import org.encog.persist.EncogDirectoryPersistence;
import org.encog.util.arrayutil.NormalizationAction;
import org.encog.util.arrayutil.NormalizeArray;
import org.encog.util.arrayutil.NormalizedField;
import org.encog.util.csv.CSVFormat;

public class Main {
    // Expected for running certain methods
    private static final String XOREX = "XORExample";
    private static final String LUNEX = "LunarExample";
    private static final String MEMEX = "MemArrExample";
    private static final String FILEX = "FileExample";
    // Propogation types
    private static final String resilientPropogation = "RESILIENT"; // DEFAULT
    private static final String backPropogation = "BACK";
    private static final String manhattanPropogation = "MAN";
    private static final String quickPropagation = "QPROP";

    private static String propogation = resilientPropogation;

    public static void main(String[] args) {
        // Determine if propogation argument applied
        String propArg = args[0];
        if (propArg != null) {
            propogation = propArg.split("=")[1];
        }
        String runProp = System.getProperty("run");
        if (runProp.contains(",")) {
            // Split it up
            List<String> methodsToRun = Arrays.asList(runProp.split(","));
            // Run what's in the list
            for (String m : methodsToRun) {
                runMethod(m);
            }
        } else if (runProp.trim().equalsIgnoreCase("help")) {
            runHelp(null );
        }
        else {
            runMethod(runProp);
        }
    }

    private static void runMethod(String m) {
        switch (m) {
            case XOREX:
                runXORExample();
                break;
            case LUNEX:
                runLunarNormalExample();
                break;
            case MEMEX:
                runMemArrayNormalExample();
                break;
            case FILEX:
                runFileNormalExample();
                break;
            default:
                runHelp(m);
                break;
        }
    }

    private static void runXORExample() {
        System.out.println("Running " + Main.class.getSimpleName() + ".runXORExample(PROPOGATION='" + propogation +"')");
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
        Propagation trainer = null;
        double learningRate;

        if (propogation.equalsIgnoreCase(resilientPropogation)) {
            trainer = new ResilientPropagation(network, trainingSet);
        } else if (propogation.equalsIgnoreCase(backPropogation)) {
            // Using backpropagation training for this example
            learningRate = 0.7;
            double momentum = 0.3;
            trainer = new Backpropagation(network, trainingSet, learningRate, momentum);
        } else if (propogation.equalsIgnoreCase(manhattanPropogation)) {
            learningRate = 0.00001;
            trainer = new ManhattanPropagation(network, trainingSet, learningRate);
        } else if (propogation.equalsIgnoreCase(quickPropagation)) {
            learningRate = 2.0;
            trainer = new QuickPropagation(network, trainingSet, learningRate);
        }
        else {
            throw new IllegalArgumentException("Unexpected propogation: " + propogation);
        } 
        
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
        System.out.println("Network error: " + network.calculateError(trainingSet));
        // Example practices persistence
        String nnFile = System.getProperty("user.dir") + File.separator + "networks" + File.separator + "xorNN.eg";
        EncogDirectoryPersistence.saveObject(new File(nnFile), network);
        System.out.println("Network saved to " + nnFile);
        // Load the network back in a compare (should be the same)
        BasicNetwork loadedNetwork = (BasicNetwork) EncogDirectoryPersistence.loadObject(new File(nnFile));
        System.out.println("Loaded network error: " + loadedNetwork.calculateError(trainingSet));
    }

    private static void runLunarNormalExample() {
        // Create normalized field for fuel stats
        // Acceptable range (0-200)
        // Normalization range (-0.9-0.9)
        System.out.println("Running " + Main.class.getSimpleName() + ".runLunarNormalExample()");
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
        System.out.println("Running " + Main.class.getSimpleName() + ".runMemArrayNormalExample()");
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
    
    private static void runFileNormalExample() {
        // The file we need to normalize
        System.out.println("Running " + Main.class.getSimpleName() + ".runFileNormalExample()");
        File irsFile = new File(System.getProperty("user.dir") + 
                File.separator + 
                "datasets" + 
                File.separator +
                "iris.csv");
        File resultFile = new File(System.getProperty("user.dir") +
                File.separator +
                "outputs" +
                File.separator +
                "iris_normalized.csv");

        EncogAnalyst encogAnalyst = new EncogAnalyst();
        // The wizard reads the source file and builds normalization stats
        AnalystWizard analystWizard = new AnalystWizard(encogAnalyst);
        // Start the wizard
        analystWizard.wizard(irsFile, true, AnalystFileFormat.DECPNT_COMMA);
        // Now, create the normalization object
        final AnalystNormalizeCSV normalizeCSV = new AnalystNormalizeCSV();
        normalizeCSV.analyze(irsFile, true, CSVFormat.ENGLISH, encogAnalyst);
        normalizeCSV.setProduceOutputHeaders(true);
        // Normalize the file and put results in result file
        normalizeCSV.normalize(resultFile);
        System.out.println("Results found in " + resultFile.getAbsolutePath());
    }

    private static void runHelp(String m) {
        if (m != null) {
            System.out.println("No such method " + m + " exists.");
        }
        System.out.println("Progam usage");
        System.out.println("-Drun: System property with name of example to execute. This could be a single name or a list of names separated by comma (EX: -Drun=XORExample,LunarExample)");
        System.out.println("EXAMPLE: java -jar -Drun=XORExample target/NNLearning-1.0-SNAPSHOT.jar");
    }
}
