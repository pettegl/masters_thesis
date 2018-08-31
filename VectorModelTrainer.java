
import org.deeplearning4j.models.embeddings.learning.impl.elements.SkipGram;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.sentenceiterator.LineSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentencePreProcessor;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;

import java.io.File;

/**
 * Created by Petter Glad-Ørbak.
 */
public class VectorModelTrainer {

    // Path to text file for training the model.
    public static final String PATH_TO_TRAINING_CORPUS = "D:/Master/gen/tagsfile25k.txt";

    // Path to save the Word2vec model after training.
    public static final String PATH_TO_SAVE_W2V_MODEL = "NewExperiment.zip";

    public static void main(String[] args) {
        try {

            SentenceIterator iter = new LineSentenceIterator(new File(PATH_TO_TRAINING_CORPUS));
            iter.setPreProcessor(new SentencePreProcessor()
            {
                @Override
                public String preProcess(String sentence)
                {
                    return sentence.toLowerCase();
                }
            });

            TokenizerFactory tf = new DefaultTokenizerFactory();
            tf.setTokenPreProcessor(new CommonPreprocessor());


            Word2Vec vec = new Word2Vec.Builder()
                    // Discard words that occur less than 3 times
                    .minWordFrequency(6)
                    // Number of iterations over each sentence when training
                    .iterations(1)
                    // Number of vector dimensions
                    .layerSize(300)
                    // Number of iterations over corpus when training
                    .epochs(30)
                    // Learning algorithm. Skip-gram is used here, but new CBOW<VocabWord>() would invoke CBOW.
                    .elementsLearningAlgorithm(new SkipGram<VocabWord>())
                    // Learning rate
                    .learningRate(0.05)
                    // Seed for random number generator
                    .seed(42)
                    // Context window size. Mikolov et al. recommend between 5-10.
                    .windowSize(8)
                    // Set the sentence iterator
                    .iterate(iter)
                    .tokenizerFactory(tf)
                    .build();
            // Train
            vec.fit();

            // Write Word2vec model to path.
            WordVectorSerializer.writeWord2VecModel(vec, PATH_TO_SAVE_W2V_MODEL);



        } catch (Exception e)
        {
            System.out.println("Error occured: " + e);
            e.printStackTrace();
        }
    }


}
