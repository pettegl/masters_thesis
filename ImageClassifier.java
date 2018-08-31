
import org.tensorflow.*;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * Created by Petter Glad-Ørbak.
 */
public class ImageClassifier
{
    // Path of the pre-trained model to use for classification.
    private static final String MODEL_DIR = "D:/Master/src/main/resources/models/";

    // Path to image folder to classify.
    private String IMG_PATH;

    // Pre-trained model read as bytes.
    private byte[] GRAPH_DEF;

    // Classification labels.
    private List<String> LABELS;

    // Graph for Tensor computation.
    private Graph tf_g;

    // Tensorflow session to run the graph on.
    private Session tf_s;

    private BufferedWriter BW;

    public ImageClassifier(String path)
    {
        try {

            // Path to write predictions to
            BW = new BufferedWriter(new FileWriter("top1predictions.txt"));

            // Path of images to classify
            IMG_PATH = path;

            // Path to pre-trained model Inception-v3 saved as a Graph (.pb).
            GRAPH_DEF = readAllBytesOrExit(Paths.get(MODEL_DIR, "tensorflow_inception_graph.pb"));

            // Path to synset of 1000 classification categories.
            LABELS = readAllLinesOrExit(Paths.get(MODEL_DIR, "imagenet_comp_graph_label_strings.txt"));

            // Import saved Inception-v3 computation graph
            tf_g = new Graph();
            tf_g.importGraphDef(GRAPH_DEF);
            tf_s = new Session(tf_g);

            // Classify images and write predictions to file.
            writePredictionsToFile();


        } catch (Exception e)
        {
            System.out.println("Error occured: " + e);
            e.printStackTrace();
        }
    }
    /*
    Return a text file as a List<String> where each line of text is an object.
     */
    private List<String> readAllLinesOrExit(Path path) {
        try {
            return Files.readAllLines(path, Charset.forName("UTF-8"));
        } catch (IOException e) {
            System.err.println("Failed to read [" + path + "]: " + e.getMessage());
            System.exit(0);
        }
        return null;
    }
    /*
    Return a byte array with the contents of Path path.
     */
    private byte[] readAllBytesOrExit(Path path) {
        try {
            return Files.readAllBytes(path);
        } catch (IOException e) {
            System.err.println("Failed to read [" + path + "]: " + e.getMessage());
            System.exit(1);
        }
        return null;
    }

    private Session getSession()
    {
        return this.tf_s;
    }
    /*
    Return a list of Path objects to all the files in a given folder.
     */
    private List<Path> getPaths()
    {
        File[] files = new File(IMG_PATH).listFiles();

        List<Path> pathList = new ArrayList<>();

        for (File file : files) {
            if (file.isFile() && !pathList.contains(Paths.get(file.getAbsolutePath()))) {
                pathList.add(Paths.get(file.getAbsolutePath()));
            }
        }
        return pathList;
    }

    /*
    From a list of paths, read image bytes and convert it to a Tensor. Execute the tensor and obtain
    predictions and their probabilities. Do a light preprocessing and write the top five predictions as
    one line in a file.
     */
    public void writePredictionsToFile()
    {
        try {

            List<Path> pathList = getPaths();

            for (Path p : pathList) {

                byte[] imageBytes = readAllBytesOrExit(p);

                System.out.println(p.getFileName());

                Tensor image = Tensor.create(imageBytes);

                float[] labelProbabilities = executeTensorOnGraph(image, this.tf_s);

                List<Float> probsArray = new ArrayList<>();

                for (float f : labelProbabilities) {
                    probsArray.add(f);
                }

                Collections.sort(probsArray);

                Collections.reverse(probsArray);


                List<String> topFive = new ArrayList<>();

                // i < number of top predictions you want to test.

                for (int i = 0; i < 1; i++) {
                    if (getIndex(labelProbabilities, probsArray.get(i)) != -1) {

                        if(probsArray.get(i) >= 0.01) {
                            topFive.add(LABELS.get(getIndex(labelProbabilities, probsArray.get(i))));

                            // Optional line to print predictions with probabilities to stdout as they happen
//                        System.out.println(topFive.get(i) + " with probability: " + probsArray.get(i));

                        if (topFive.get(i).contains("'")) {
                            topFive.set(i, (topFive.get(i).replace(("'"), " ")));
                        }
                        if (topFive.get(i).contains("-")) {
                                topFive.set(i, (topFive.get(i).replace(("-"), " ")));
                        }
                        if (topFive.get(i).contains(" ")) {
                            topFive.set(i, (topFive.get(i).replace((" "), " ")));
                        }
                            BW.write(topFive.get(i) + " ");
                        }


                    } else {
                        throw new Exception("Couldn't find index for probability label.");
                    }
                }
                BW.newLine();

            }
            BW.close();

        } catch (Exception e) {
            e.printStackTrace();
        }

    }
    /*
    Execute tensor on graph, and obtain float array of probabilities and the their belonging label indexes.
    This is a modified function from TensorFlow's own example "LabelImage.java".
     */
    private float[] executeTensorOnGraph(Tensor image, Session s) {
        try {

            Tensor result = s.runner().feed("DecodeJpeg/contents", image).fetch("softmax").run().get(0);
            final long[] rshape = result.shape();

            if (result.numDimensions() != 2 || rshape[0] != 1) {
                throw new RuntimeException(
                        String.format(
                                "Expected model to produce a [1 N] shaped tensor where N is the number of labels, instead it produced one with shape %s",
                                Arrays.toString(rshape)));
            }

            int nlabels = (int) rshape[1];
            return result.copyTo(new float[1][nlabels])[0];

        } catch (Exception e) { e.printStackTrace(); }

        return null;
    }
    /*
    Return index of probability.
     */
    private Integer getIndex(float[] probs, float prob)
    {
        for(int i = 0; i < probs.length; i++)
        {
            if(probs[i] == prob)
            {
                return i;
            }
        }
        return -1;
    }


}
