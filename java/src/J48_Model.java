import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.Evaluation;

import java.io.File;
import java.util.Random;

public class J48_Model {

    public static void main(String[] args) {
        try {
            // Input CSV file path
            String csvFilePath = "../data/.csv/train_data.csv";

            // Convert CSV to ARFF
            String arffFilePath = convertCSVToArff(csvFilePath);
            if (arffFilePath == null) {
                System.out.println("Failed to convert CSV to ARFF.");
                return;
            }

            // Load the ARFF dataset
            DataSource source = new DataSource(arffFilePath);
            Instances data = source.getDataSet();

            // Ensure the class attribute is set
            data.setClassIndex(data.numAttributes() - 1);

            // Shuffle the dataset for randomness
            data.randomize(new Random());

            // Define the train-test split ratio
            double trainPercentage = 70.0; // Set your desired train percentage
            int trainSize = (int) Math.round(data.numInstances() * trainPercentage / 100);
            int testSize = data.numInstances() - trainSize;

            // Split the data
            Instances trainSet = new Instances(data, 0, trainSize);
            Instances testSet = new Instances(data, trainSize, testSize);

            // Print dataset sizes
            System.out.println("Original dataset size: " + data.numInstances());
            System.out.println("Training dataset size: " + trainSet.numInstances());
            System.out.println("Testing dataset size: " + testSet.numInstances());

            // Save train and test datasets
            saveInstances(trainSet, "../data/.arff/train_data.arff");
            saveInstances(testSet, "../data/.arff/test_data.arff");

            // Train and evaluate the model
            trainAndEvaluate(trainSet, testSet);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    // Method to convert CSV to ARFF and save it in a specified directory structure
    public static String convertCSVToArff(String csvFilePath) {
        try {
            // Replace ".csv" with ".arff" and change the directory to "../data/.arff/"
            String arffFilePath = csvFilePath.replace("/.csv/", "/.arff/").replace(".csv", ".arff");

            // Load CSV file
            CSVLoader loader = new CSVLoader();
            loader.setSource(new File(csvFilePath));
            Instances data = loader.getDataSet();

            // Save data to ARFF file
            ArffSaver saver = new ArffSaver();
            saver.setInstances(data);
            saver.setFile(new File(arffFilePath));
            saver.writeBatch();

            System.out.println("CSV converted to ARFF and saved at: " + arffFilePath);
            return arffFilePath; // Return the generated ARFF file path
        } catch (Exception e) {
            e.printStackTrace();
            return null; // Return null in case of failure
        }
    }

    // Method to save Instances to a file with incremented suffix if file exists
    public static void saveInstances(Instances data, String filePath) throws Exception {
        File file = new File(filePath);
        String baseName = file.getName();
        String parentDir = file.getParent();
        String fileNameWithoutExt = baseName.contains(".") ? baseName.substring(0, baseName.lastIndexOf("."))
                : baseName;
        String extension = baseName.contains(".") ? baseName.substring(baseName.lastIndexOf(".")) : "";
        int counter = 0;

        // Increment the filename if the file already exists
        while (file.exists()) {
            counter++;
            String newFileName = fileNameWithoutExt + "(" + counter + ")" + extension;
            file = new File(parentDir, newFileName);
        }

        // Save the Instances to the newly created or modified file path
        ArffSaver saver = new ArffSaver();
        saver.setInstances(data);
        saver.setFile(file);
        saver.writeBatch();

        System.out.println("File saved successfully at: " + file.getAbsolutePath());
    }

    // Method to train and evaluate the model
    public static void trainAndEvaluate(Instances trainSet, Instances testSet) throws Exception {
        // Choose the classifier
        Classifier classifier = new J48();

        // Train the classifier
        classifier.buildClassifier(trainSet);

        // Evaluate the model
        Evaluation evaluation = new Evaluation(trainSet);
        evaluation.evaluateModel(classifier, testSet);

        // Print evaluation results
        System.out.printf("Overall Accuracy: %.2f%%\n", (evaluation.correct() / evaluation.numInstances()) * 100);

        // Print class-specific metrics
        int numClasses = testSet.numClasses();
        for (int i = 0; i < numClasses; i++) {
            System.out.println("\nClass: " + testSet.classAttribute().value(i));
            System.out.printf("Precision: %.2f\n", evaluation.precision(i));
            System.out.printf("Recall: %.2f\n", evaluation.recall(i));
            System.out.printf("F1-Score: %.2f\n", evaluation.fMeasure(i));
        }

        // Print Confusion Matrix
        System.out.println("\nConfusion Matrix:");
        double[][] confusionMatrix = evaluation.confusionMatrix();
        for (int i = 0; i < numClasses; i++) {
            for (int j = 0; j < numClasses; j++) {
                System.out.printf("%.0f\t", confusionMatrix[i][j]);
            }
            System.out.println();
        }
    }
}
