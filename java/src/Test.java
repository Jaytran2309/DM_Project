import weka.classifiers.meta.CVParameterSelection;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.Evaluation;

import java.util.Random;

public class Test {
  public static void main(String[] args) {
    try {
      // Load dataset
      DataSource source = new DataSource("../data/.arff/full_data.arff"); // Replace with your ARFF file path
      Instances data = source.getDataSet();

      // Set the class index
      data.setClassIndex(data.numAttributes() - 1);

      // Initialize CVParameterSelection
      CVParameterSelection cvParamSel = new CVParameterSelection();
      RandomForest rf = new RandomForest();
      rf.setNumExecutionSlots(10); // Set num-slots to 10 for parallel processing
      cvParamSel.setClassifier(rf);

      // Add parameters to be tuned
      // Number of Trees (I): Tune from 50 to 200 in steps of 50
      cvParamSel.addCVParameter("I 10 100 5");

      // Maximum Depth (depth): Tune from 0 (unlimited) to 20 in steps of 5
      cvParamSel.addCVParameter("depth 0 20 5");

      // Minimum Number of Instances Per Leaf (M): Tune from 1 to 10 in steps of 1
      cvParamSel.addCVParameter("M 1 10 1");

      // Variance Split Threshold (V): Tune from 0.0 to 0.1 in steps of 0.02
      cvParamSel.addCVParameter("V 0.0 0.1 0.02");

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
