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
  var sc: SparkContext = _
  var model: Word2Vec = _

  override def beforeAll(): Unit = {
    // Initialize Spark context and Word2Vec model for testing
    sc = SlidingWindowTraining.createSparkContext()
    model = WordVectorSerializer.readWord2VecModel("src/main/resources/wordvectors.txt")
  }

  override def afterAll(): Unit = {
    sc.stop()
  }

  // This test is used to check the creation of a SparkContext, and that it is running locally for testing
  test("createSparkContext() should initialize a local Spark context") {
    assert(sc != null)
    assert(sc.isLocal, "Spark context should be in local mode for testing")
  }

  test("createSlidingWindows() should generate sliding windows and targets") {
    val sentence = Array("hello", "this", "is", "a", "test", "for", "scala")

    // Setting hard values only for testing with small file
    val windowSize = 2 // config.getInt("app.windowSize")
    val overlapSize = 1 // config.getInt("app.overlapSize")
    val embeddingDimensions = 3   // This number has to match dimension in wordvectors.txt
    val dataset: DataSet = SlidingWindowTraining.createSlidingWindows(sentence, windowSize, overlapSize, model)

    assert(dataset != null)
    assert(dataset.getFeatures.rank() == 3, "Input features should be a 3D tensor")

    // 6 windows, 3 embedding dimensions, and 2 words per window
    assert(dataset.getFeatures.shape().sameElements(Array(6, embeddingDimensions,
      windowSize)), "The shape of the features should be [batchSize, embeddingSize, SequenceLength]")

    assert(dataset.getLabels.rank() == 2, "Output labels should be a 2D tensor")
  }

  test("getTrainingData() should return a DataSet with correct shapes") {
    val inputVectors = Array(Array(Array(1.0, 0.5, 0.1), Array(0.2, 0.3, 0.6)))
    val outputVectors = Array(Array(0.1, 0.2, 0.3))
    val dataset: DataSet = SlidingWindowTraining.getTrainingData(inputVectors, outputVectors)

    assert(dataset != null)
    assert(dataset.getFeatures.shape().sameElements(Array(1, 3, 2)), "Feature shape should match input vectors")
    assert(dataset.getLabels.shape().sameElements(Array(1, 3)), "Label shape should match output vectors")
  }

  test("createMultiLayerConfiguration() should configure a neural network model") {
    val config = SlidingWindowTraining.createMultiLayerConfiguration()
    assert(config != null, "Configuration should not be null")
    //assert(config.getConf(0).getLayer.getNIn == 3, "First layer input should match word vector size")
  }

  test("countWindows() should correctly calculate the number of sliding windows") {
    val n = 10
    val w = 3
    val o = 1
    val result = SlidingWindowTraining.countWindows(n, w, o)
    assert(result == 5, s"Expected 4 windows, but got $result")
  }
}
