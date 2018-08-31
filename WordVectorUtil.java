
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import java.util.Collection;


/**
 * Created by Petter Glad-Ørbak.
 */
public class WordVectorUtil
{

    public Word2Vec vec;

    // Load the word2vec model from path provided in constructor.
    public WordVectorUtil(String pathToModel)
    {
        vec = WordVectorSerializer.readWord2VecModel(pathToModel);
    }
    /*
    Return the 'numberOfNearest' neighbors of a given word as a lower case string of tokens. Check for duplicates.
     */
    public String getNearestWordsAsString(String word, int numberOfNearest)
    {
        String neighborString = "";

        // Query model for 'numberOfNearest' vectors of closest cosine similarity
        Collection<String> nearestNeighbors = vec.wordsNearest(word, numberOfNearest);

        for(String neighbor : nearestNeighbors)
        {
            if(!neighborString.contains(neighbor) && !neighbor.equalsIgnoreCase(word)) neighborString += neighbor + " ";
        }
        return neighborString;

    }
    public Word2Vec getVec()
    {
        return vec;
    }

}
