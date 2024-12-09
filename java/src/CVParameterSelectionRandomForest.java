import weka.classifiers.meta.CVParameterSelection;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.Evaluation;

import java.util.Random;

public class CVParameterSelectionRandomForest {
    public static void main(String[] args) {
        try {
            // Load dataset
            DataSource source = new DataSource("../data/.arff/data910.arff"); // Replace with your
                                                                                                   // ARFF file path
            Instances data = source.getDataSet();

            // Set the class index
            data.setClassIndex(data.numAttributes() - 1);

            // Initialize CVParameterSelection
            CVParameterSelection cvParamSel = new CVParameterSelection();
            RandomForest rf = new RandomForest();
            rf.setNumExecutionSlots(1); // Set num-slots to 10 for parallel processing
            cvParamSel.setClassifier(rf);

            // Add parameters to be tuned
            cvParamSel.addCVParameter("I 10 100 10"); // Number of trees (10 to 100, step 10)
            cvParamSel.addCVParameter("K 0 5 1"); // Number of attributes to consider (0 to 5, step 1)
            cvParamSel.addCVParameter("depth 5 25 5"); // Max depth (5 to 25, step 5)

            // Perform cross-validation parameter selection
            cvParamSel.setNumFolds(10); // 10-fold cross-validation
            cvParamSel.buildClassifier(data);

            // Output the selected parameters
            System.out.println("Best Parameters Found:");
            String[] bestOptions = cvParamSel.getBestClassifierOptions();
            for (String option : bestOptions) {
                System.out.print(option + " ");
            }
            System.out.println();

            // Build the best classifier with the best options
            RandomForest bestClassifier = new RandomForest();
            bestClassifier.setOptions(bestOptions);
            bestClassifier.buildClassifier(data);

            // Evaluate the best classifier using cross-validation
            Evaluation eval = new Evaluation(data);
            eval.crossValidateModel(bestClassifier, data, 10, new Random(1));

            // Print evaluation results
            System.out.printf("\nOverall Accuracy: %.2f%%\n", eval.pctCorrect());
            System.out.println("\n=== Detailed Accuracy By Class ===");
            System.out.println("TP Rate\tFP Rate\tPrecision\tRecall\tF-Measure\tROC Area\tClass");

            int numClasses = data.numClasses();
            for (int i = 0; i < numClasses; i++) {
                System.out.printf(
                        "%.3f\t%.3f\t%.3f\t\t%.3f\t%.3f\t\t%.3f\t\t%s\n",
                        eval.truePositiveRate(i),
                        eval.falsePositiveRate(i),
                        eval.precision(i),
                        eval.recall(i),
                        eval.fMeasure(i),
                        eval.areaUnderROC(i),
                        data.classAttribute().value(i));
            }

            // Confusion matrix
            System.out.println("\n=== Confusion Matrix ===");
            double[][] confusionMatrix = eval.confusionMatrix();
            for (double[] row : confusionMatrix) {
                for (double value : row) {
                    System.out.printf("%.0f\t", value);
                }
                System.out.println();
            }

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
