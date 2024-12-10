import weka.core.converters.CSVLoader;
import weka.core.converters.ArffSaver;
import weka.core.Instances;
import java.io.File;

public class CSVToARFF {
    public static void main(String[] args) {
        try {
            // Paths to the CSV and ARFF files relative to "java/src/"
            String name ="validation";
            String csvFilePath = "../data/.csv/"+ name +".csv"; // Input CSV file
            String arffFilePath = "../data/.arff/"+ name +".arff"; // Output ARFF file

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
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
