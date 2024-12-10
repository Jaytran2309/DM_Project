import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

import java.io.FileWriter;
import java.util.Random;

import org.json.JSONArray;
import org.json.JSONObject;

public class RandomForest_Model {

    public static void main(String[] args) {
        try {
            // Hard-coded relative paths for train and test datasets
            String trainFilePath = "../data/.arff/combined_train_validation.arff";
            String testFilePath = "../data/.arff/test.arff";

            // Load train and test datasets
            Instances trainSet = loadDataset(trainFilePath);
            Instances testSet = loadDataset(testFilePath);

            // Ensure the class attribute is set
            trainSet.setClassIndex(trainSet.numAttributes() - 1);
            testSet.setClassIndex(testSet.numAttributes() - 1);

            // Get RandomForest classifier with customizable parameters
            RandomForest randomForest = getRandomForestClassifier();

            // Train and evaluate the model
            Evaluation evaluation = trainAndEvaluate(randomForest, trainSet, testSet);

            // Print evaluation metrics
            printEvaluationMetrics(evaluation, testSet);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static Instances loadDataset(String filePath) throws Exception {
        DataSource source = new DataSource(filePath);
        return source.getDataSet();
    }

    public static RandomForest getRandomForestClassifier() throws Exception {
        RandomForest randomForest = new RandomForest();
        // Customize hyperparameters here
        String[] options = weka.core.Utils.splitOptions("-I 100 -K 0 -depth 0 -num-slots 1"); 
        // -I: number of trees, -K: number of features, -depth: max depth, -num-slots: threads
        randomForest.setOptions(options);
        return randomForest;
    }

    public static Evaluation trainAndEvaluate(Classifier classifier, Instances trainSet, Instances testSet)
            throws Exception {
        // Train the classifier
        classifier.buildClassifier(trainSet);

        // Evaluate the model
        Evaluation evaluation = new Evaluation(trainSet);
        evaluation.evaluateModel(classifier, testSet);
        saveEvaluationResults(evaluation, testSet, "RandomForest");
        return evaluation;
    }

    public static void printEvaluationMetrics(Evaluation evaluation, Instances testSet) throws Exception {
        System.out.println();
        System.out.println("=== Detailed Accuracy By Class ===");
        System.out.println("TP Rate\tFP Rate\tPrecision\tRecall\tF-Measure\tMCC\tROC Area\tPRC Area\tClass");

        double totalInstances = evaluation.numInstances();
        double weightedMCC = 0.0;
        double totalWeight = 0.0;

        for (int i = 0; i < testSet.numClasses(); i++) {
            double classWeight = evaluation.truePositiveRate(i) + evaluation.falseNegativeRate(i);
            double classMCC = evaluation.matthewsCorrelationCoefficient(i);

            weightedMCC += classWeight * classMCC;
            totalWeight += classWeight;

            System.out.printf(
                    "%.3f\t%.3f\t%.3f\t\t%.3f\t%.3f\t\t%.3f\t%.3f\t\t%.3f\t\t%s\n",
                    evaluation.truePositiveRate(i),
                    evaluation.falsePositiveRate(i),
                    evaluation.precision(i),
                    evaluation.recall(i),
                    evaluation.fMeasure(i),
                    classMCC,
                    evaluation.areaUnderROC(i),
                    evaluation.areaUnderPRC(i),
                    testSet.classAttribute().value(i));
        }

        // Normalize Weighted MCC
        weightedMCC = (totalWeight != 0) ? weightedMCC / totalWeight : 0;

        // Print Weighted Averages
        System.out.println("\n=== Weighted Averages ===");
        System.out.printf(
                "%.3f\t%.3f\t%.3f\t\t%.3f\t%.3f\t\t%.3f\t%.3f\t\t%.3f\t\tWeighted Avg.\n",
                evaluation.weightedTruePositiveRate(),
                evaluation.weightedFalsePositiveRate(),
                evaluation.weightedPrecision(),
                evaluation.weightedRecall(),
                evaluation.weightedFMeasure(),
                weightedMCC,
                evaluation.weightedAreaUnderROC(),
                evaluation.weightedAreaUnderPRC());

        // Print Confusion Matrix
        System.out.println("\n=== Confusion Matrix ===");
        double[][] confusionMatrix = evaluation.confusionMatrix();
        for (int i = 0; i < confusionMatrix.length; i++) {
            for (int j = 0; j < confusionMatrix[i].length; j++) {
                System.out.printf("%.0f\t", confusionMatrix[i][j]);
            }
            System.out.println();
        }

        // Print Overall Accuracy
        System.out.printf("\nOverall Accuracy: %.2f%%\n", evaluation.pctCorrect());
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
}
