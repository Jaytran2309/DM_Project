import weka.attributeSelection.ASEvaluation;
import weka.attributeSelection.ASSearch;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.Ranker;
import weka.attributeSelection.AttributeSelection;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import java.util.Arrays;

public class FeatureSelectionWithNaiveBayes {
    public static void main(String[] args) {
        try {
            // Step 1: Load Dataset
            String arffFilePath = "../data/.arff/full_data.arff"; // Replace with your ARFF file
            DataSource source = new DataSource(arffFilePath);
            Instances data = source.getDataSet();

            // Step 2: Set Class Index (Last Attribute as Class Label)
            data.setClassIndex(data.numAttributes() - 1);

            // Step 3: Set up Feature Selection (Evaluator + Search Method)
            ASEvaluation evaluator = new InfoGainAttributeEval(); // Information Gain Evaluator
            ASSearch search = new Ranker(); // Rank Attributes
            ((Ranker) search).setNumToSelect(5); // Select top 5 attributes (adjust as needed)

            // Step 4: Perform Attribute Selection
            AttributeSelection attributeSelector = new AttributeSelection();
            attributeSelector.setEvaluator(evaluator);
            attributeSelector.setSearch(search);
            attributeSelector.SelectAttributes(data); // Perform the selection

            // Get the indices of the selected attributes
            int[] selectedIndices = attributeSelector.selectedAttributes();

            // Step 5: Print the names of the selected features
            System.out.println("Selected Features:");
            for (int index : selectedIndices) {
                // Exclude the class attribute from the list of selected features
                if (index == data.classIndex())
                    continue;
                System.out.println(data.attribute(index).name());
            }

            // Create a new dataset with selected attributes
            Instances filteredData = attributeSelector.reduceDimensionality(data);

            // Step 6: Train Naive Bayes on Filtered Data
            Classifier naiveBayes = new NaiveBayes();
            naiveBayes.buildClassifier(filteredData);

            // Step 7: Cross-Validate Model
            weka.classifiers.Evaluation evaluation = new weka.classifiers.Evaluation(filteredData);
            evaluation.crossValidateModel(naiveBayes, filteredData, 10, new java.util.Random(1));

            // Print Evaluation Metrics
            System.out.println(evaluation.toSummaryString("Evaluation Results:\n", false));
            System.out.println("Detailed Accuracy By Class:\n" + evaluation.toClassDetailsString());
            System.out.println("Confusion Matrix:\n" + Arrays.deepToString(evaluation.confusionMatrix()));

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
