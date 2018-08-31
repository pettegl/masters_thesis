/**
 * Created by Petter Glad-Ørbak on 18.04.2018.
 */

import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Scanner;

public class W2vTest
{

    public static void main(String[] args)
    {
        try {

            Scanner sc = new Scanner(System.in);

            Word2Vec vec = WordVectorSerializer.readWord2VecModel("D:/Master/w2v/models/GoogleNews-vectors-negative300-SLIM.bin" +
                    "");

            System.out.println("Number of words in vocabulary: " + vec.getVocab().numWords());

            String s;

            do
            {
                System.out.println("Input word to find nearest 10 words: ");

                s = sc.nextLine();

                double time1 = System.currentTimeMillis();
                Collection<String> list = vec.wordsNearest(s, 10);
                double time2 = System.currentTimeMillis();
                System.out.println("Lookup time: " + (time2-time1));

                String output = "Nearest 10 words of " + s + ": " + Arrays.toString(list.toArray());

                System.out.println(output);

            } while (!s.equals("EXITPROGRAM"));

            WordVectorSerializer.writeWord2VecModel(vec, "D:/Master/w2v/models/TulleModel.zip");




        } catch (Exception e)
        {
            e.printStackTrace();
        }


    }



}
