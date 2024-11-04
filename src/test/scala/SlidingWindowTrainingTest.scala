import SlidingWindowTraining.config
import com.typesafe.config.ConfigFactory
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.BeforeAndAfterAll
import org.apache.spark.SparkContext
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer
import org.deeplearning4j.models.word2vec.Word2Vec
import org.nd4j.linalg.dataset.DataSet
import org.slf4j.LoggerFactory

class SlidingWindowTrainingTest extends AnyFunSuite with BeforeAndAfterAll {

  private val config = ConfigFactory.load()

  // These are vars that are lazily evaluated, since they are initialized every time one of the tests
  // in this file is run. These vars are not meant to be accessed in a parallelized way, so there shouldn't be any
  // problem with them, and it's safe to use them in this instance.
  var sc: SparkContext = _
  var model: Word2Vec = _

  // Setting up SparkContext and reading Word2Vec model for testing
  override def beforeAll(): Unit = {
    // Initialize Spark context and Word2Vec model for testing
    sc = SlidingWindowTraining.createSparkContext("local[*]")
    model = WordVectorSerializer.readWord2VecModel("src/main/resources/word-vectors.txt")
  }

  // Stop SparkContext after the tests
  override def afterAll(): Unit = {
    sc.stop()
  }

  // This test is used to check the creation of a SparkContext, and that it is running locally for testing
  test("createSparkContext() should initialize a local Spark context") {
    assert(sc != null)
    assert(sc.isLocal, "Spark context should be in local mode for testing")
  }

  // This test checks that createSlidingWindows() creates the sliding windows correctly with their targets
  test("createSlidingWindows() should generate sliding windows and targets") {
    val sentence = Array("hello", "this", "is", "a", "test", "for", "scala", "to", "check", "windows")

    // Setting hard values only for testing with small file
    val windowSize = 3 // config.getInt("test.windowSize")
    val overlapSize = 1 // config.getInt("test.overlapSize")
    val embeddingDimensions = 3   // This number has to match dimension in word-vectors.txt
    val dataset: DataSet = SlidingWindowTraining.createSlidingWindows(sentence, windowSize, overlapSize, model, config.getInt("test.embeddingDimensions"))

    assert(dataset != null)
    assert(dataset.getFeatures.rank() == 3, "Input features should be a 3D tensor") // checking rank of input

    // 4 windows, 3 embedding dimensions, and 3 words per window
    assert(dataset.getFeatures.shape().sameElements(Array(4, embeddingDimensions,
      windowSize)), "The shape of the features should be [batchSize, embeddingSize, SequenceLength]")

    assert(dataset.getLabels.rank() == 2, "Output labels should be a 2D tensor")  // checking rank ot output labels
  }

  // This test checks the method getTrainingData() and that it generates the correct input format for the model
  test("getTrainingData() should return a DataSet with correct shapes") {
    val inputVectors = Array(Array(Array(1.0, 0.5, 0.1), Array(0.2, 0.3, 0.6)))
    val outputVectors = Array(Array(0.1, 0.2, 0.3))
    val dataset: DataSet = SlidingWindowTraining.getTrainingData(inputVectors, outputVectors)

    assert(dataset != null)
    assert(dataset.getFeatures.shape().sameElements(Array(1, 3, 2)), "Feature shape should match input vectors")
    assert(dataset.getLabels.shape().sameElements(Array(1, 3)), "Label shape should match output vectors")
  }

  // This test checks that the multilayer configuration is being created properly for training
  test("createMultiLayerConfiguration() should configure a neural network model") {
    val configNN = SlidingWindowTraining.createMultiLayerConfiguration(config.getInt("test.embeddingDimensions"))
    assert(configNN != null, "Configuration should not be null")
  }

  // This test checks that the function to count how many windows a sentence will have based on the number of
  // words in the sentence, the window size and the overlap size
  test("countWindows() should correctly calculate the number of sliding windows") {
    val n = 10  // number of total elements
    val w = 3 // number of elements per window
    val o = 1 // number of overlap elements
    val result = SlidingWindowTraining.countWindows(n, w, o)
    assert(result == 5, s"Expected 5 windows, but got $result")
  }
}
