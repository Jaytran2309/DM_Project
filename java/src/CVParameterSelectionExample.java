import weka.classifiers.meta.CVParameterSelection;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.Evaluation;

import java.util.Random;

public class CVParameterSelectionExample {
    public static void main(String[] args) {
        try {
            // Load dataset
            DataSource source = new DataSource("../data/.arff/data910.arff"); // Replace with your ARFF file
                                                                                       // path
            Instances data = source.getDataSet();

            // Set the class index
            data.setClassIndex(data.numAttributes() - 1);

            // Initialize CVParameterSelection
            CVParameterSelection cvParamSel = new CVParameterSelection();
            cvParamSel.setClassifier(new J48()); // Set the base classifier (e.g., J48)

            // Add parameters to be tuned
            cvParamSel.addCVParameter("C 0.1 0.5 5"); // Tune confidence factor (C) for J48 pruning
            cvParamSel.addCVParameter("M 1 10 10"); // Tune minNumObj (M) for J48

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
            J48 bestClassifier = new J48();
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
