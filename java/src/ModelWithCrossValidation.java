import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;
import weka.core.converters.ConverterUtils.DataSource;

import org.json.JSONObject;
import org.json.JSONArray;

import java.io.File;
import java.io.FileWriter;
import java.util.Random;
import java.util.Scanner;

public class ModelWithCrossValidation {

    public static void main(String[] args) {
        try {
            String csvFilePath = "../data/.csv/full_data.csv";
            String arffFilePath = convertCSVToArff(csvFilePath);

            if (arffFilePath == null) {
                System.out.println("Failed to convert CSV to ARFF.");
                return;
            }

            Instances data = loadDataset(arffFilePath);
            data.setClassIndex(data.numAttributes() - 1);

            Scanner scanner = new Scanner(System.in);
            System.out.println("Choose evaluation method:");
            System.out.println("1. Train-Test Split");
            System.out.println("2. Cross-Validation");
            int choice = scanner.nextInt();

            System.out.println("\nChoose classifier:");
            System.out.println("1. J48 (Decision Tree)");
            System.out.println("2. Naive Bayes");
            System.out.println("3. Random Forest");
            int classifierChoice = scanner.nextInt();

            Classifier classifier = null;
            String modelName = "";
            if (classifierChoice == 1) {
                classifier = getJ48Classifier();
                modelName = "default_J48";
            } else if (classifierChoice == 2) {
                classifier = new NaiveBayes();
                modelName = "default_NaiveBayes";
            } else if (classifierChoice == 3) {
                classifier = getRandomForestClassifier();
                modelName = "default_RandomForest";
            } else {
                System.out.println("Invalid choice!");
                return;
            }

            if (choice == 1) {
                System.out.println("Enter the train percentage (e.g., 70 for 70% train, 30% test):");
                double trainPercentage = scanner.nextDouble();
                Instances[] split = splitData(data, trainPercentage);
                Instances trainSet = split[0];
                Instances testSet = split[1];

                Evaluation evaluation = trainAndEvaluate(classifier, trainSet, testSet, modelName);
                saveEvaluationResults(evaluation, testSet, modelName);
            } else if (choice == 2) {
                System.out.println("Enter the number of folds for cross-validation:");
                int numFolds = scanner.nextInt();
                Evaluation evaluation = crossValidationEvaluation(classifier, data, numFolds, modelName);
                saveEvaluationResults(evaluation, data, modelName);
            } else {
                System.out.println("Invalid choice. Exiting.");
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static String convertCSVToArff(String csvFilePath) {
        try {
            String arffFilePath = csvFilePath.replace("/.csv/", "/.arff/").replace(".csv", ".arff");

            CSVLoader loader = new CSVLoader();
            loader.setSource(new File(csvFilePath));
            Instances data = loader.getDataSet();

            ArffSaver saver = new ArffSaver();
            saver.setInstances(data);
            saver.setFile(new File(arffFilePath));
            saver.writeBatch();

            System.out.println("CSV converted to ARFF and saved at: " + arffFilePath);
            return arffFilePath;
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }

    public static Instances loadDataset(String filePath) throws Exception {
        DataSource source = new DataSource(filePath);
        return source.getDataSet();
    }

    public static J48 getJ48Classifier() throws Exception {
        J48 j48 = new J48();
        String[] options = weka.core.Utils.splitOptions("-C 0.25 -M 2"); // Default parameters
        j48.setOptions(options);
        return j48;
    }

    public static RandomForest getRandomForestClassifier() throws Exception {
        RandomForest randomForest = new RandomForest();
        String[] options = weka.core.Utils.splitOptions("-I 100 -K 0 -depth 10 -num-slots 10");
        randomForest.setOptions(options);
        return randomForest;
    }

    public static Instances[] splitData(Instances data, double trainPercentage) {
        try {
            data.randomize(new Random());
            int trainSize = (int) Math.round(data.numInstances() * trainPercentage / 100.0);
            int testSize = data.numInstances() - trainSize;

            Instances trainSet = new Instances(data, 0, trainSize);
            Instances testSet = new Instances(data, trainSize, testSize);

            saveInstances(trainSet, "../data/.arff/train_data.arff");
            saveInstances(testSet, "../data/.arff/test_data.arff");

            return new Instances[] { trainSet, testSet };
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }

    public static void saveInstances(Instances data, String filePath) throws Exception {
        File file = new File(filePath);
        if (file.exists()) {
            file.delete();
        }
        ArffSaver saver = new ArffSaver();
        saver.setInstances(data);
        saver.setFile(file);
        saver.writeBatch();
    }

    public static Evaluation trainAndEvaluate(Classifier classifier, Instances trainSet, Instances testSet,
            String modelName)
            throws Exception {
        classifier.buildClassifier(trainSet);

        // Save the trained model
        saveTrainedModel(classifier, modelName);

        Evaluation evaluation = new Evaluation(trainSet);
        evaluation.evaluateModel(classifier, testSet);

        printEvaluationMetrics(evaluation, testSet);
        return evaluation;
    }

    public static Evaluation crossValidationEvaluation(Classifier classifier, Instances data, int numFolds,
            String modelName)
            throws Exception {
        Evaluation evaluation = new Evaluation(data);
        evaluation.crossValidateModel(classifier, data, numFolds, new Random());
        printEvaluationMetrics(evaluation, data);
        return evaluation;
    }

    public static void saveTrainedModel(Classifier classifier, String modelName) {
        try {
            // Define the relative path for saving models
            String modelDirectory = "../model";
            File dir = new File(modelDirectory);

            // Create the directory if it doesn't exist
            if (!dir.exists()) {
                if (dir.mkdirs()) {
                    System.out.println("Model directory created at: " + dir.getAbsolutePath());
                } else {
                    System.err.println("Failed to create model directory at: " + dir.getAbsolutePath());
                    return;
                }
            }

            // Save the model with the given name
            String filePath = modelDirectory + "/" + modelName + ".model";
            SerializationHelper.write(filePath, classifier);

            System.out.println("Model saved successfully at: " + filePath);
        } catch (Exception e) {
            System.err.println("Failed to save the model: " + e.getMessage());
        }
    }

    public static void saveEvaluationResults(Evaluation evaluation, Instances testSet, String modelName)
            throws Exception {
        String filePath = "../data/.json/Default_Model_Results.json";

        JSONObject resultJson = new JSONObject();
        resultJson.put("Model Name", modelName);
        resultJson.put("Overall Accuracy", evaluation.pctCorrect());

        JSONArray confusionMatrixArray = new JSONArray();
        double[][] confusionMatrix = evaluation.confusionMatrix();
        for (double[] row : confusionMatrix) {
            JSONArray rowArray = new JSONArray();
            for (double value : row) {
                rowArray.put(value);
            }
            confusionMatrixArray.put(rowArray);
        }
        resultJson.put("Confusion Matrix", confusionMatrixArray);

        JSONObject classMetrics = new JSONObject();
        for (int i = 0; i < testSet.numClasses(); i++) {
            JSONObject metrics = new JSONObject();
            metrics.put("TP Rate", evaluation.truePositiveRate(i));
            metrics.put("FP Rate", evaluation.falsePositiveRate(i));
            metrics.put("Precision", evaluation.precision(i));
            metrics.put("Recall", evaluation.recall(i));
            metrics.put("F-Measure", evaluation.fMeasure(i));
            metrics.put("MCC", evaluation.matthewsCorrelationCoefficient(i));
            metrics.put("ROC Area", evaluation.areaUnderROC(i));
            metrics.put("PRC Area", evaluation.areaUnderPRC(i));
            classMetrics.put(testSet.classAttribute().value(i), metrics);
        }
        resultJson.put("Class Metrics", classMetrics);

        FileWriter file = new FileWriter(filePath, true);
        file.write(resultJson.toString(4));
        file.flush();
        file.close();

        System.out.println("Evaluation results saved at: " + filePath);
    }

    public static void printEvaluationMetrics(Evaluation evaluation, Instances testSet) throws Exception {
        System.out.println("=== Detailed Accuracy By Class ===");
        System.out.println("TP Rate\tFP Rate\tPrecision\tRecall\tF-Measure\tMCC\tROC Area\tPRC Area\tClass");

        for (int i = 0; i < testSet.numClasses(); i++) {
            System.out.printf(
                    "%.3f\t%.3f\t%.3f\t\t%.3f\t%.3f\t\t%.3f\t%.3f\t\t%.3f\t\t%s\n",
                    evaluation.truePositiveRate(i),
                    evaluation.falsePositiveRate(i),
                    evaluation.precision(i),
                    evaluation.recall(i),
                    evaluation.fMeasure(i),
                    evaluation.matthewsCorrelationCoefficient(i),
                    evaluation.areaUnderROC(i),
                    evaluation.areaUnderPRC(i),
                    testSet.classAttribute().value(i));
        }
        System.out.printf("Overall Accuracy: %.2f%%\n", evaluation.pctCorrect());
    }
}