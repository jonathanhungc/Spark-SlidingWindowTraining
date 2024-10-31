import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer
import org.deeplearning4j.nn.conf.MultiLayerConfiguration
import org.nd4j.linalg.dataset.DataSet
import org.apache.spark.rdd.RDD
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster
import org.deeplearning4j.spark.parameterserver.training.SharedTrainingMaster
import org.deeplearning4j.optimize.listeners.EvaluativeListener
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.optimize.listeners.SharedGradient
import org.deeplearning4j.optimize.listeners.PerformanceListener
import org.deeplearning4j.nn.conf.layers.{DenseLayer, GlobalPoolingLayer, LSTM, OutputLayer, PoolingType, SelfAttentionLayer}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.api.ndarray.INDArray
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.nd4j.linalg.activations.Activation
import org.deeplearning4j.models.word2vec.Word2Vec
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer
import org.slf4j.LoggerFactory
import com.typesafe.config.ConfigFactory
import org.deeplearning4j.nn.weights.WeightInit


object SlidingWindowTraining {

  private val log = LoggerFactory.getLogger("SlidingWindowTraining")
  private val config = ConfigFactory.load()

  def createSparkContext(): SparkContext = {
    // Configure Spark for local or cluster mode
    log.info("createSparkContext(): Setting up SparkConf")
    val conf = new SparkConf()
      .setAppName("Sliding Window Training")
      .setMaster("local[*]") // For local testing, or use "yarn", "mesos", etc. in a cluster

    new SparkContext(conf)
  }

  // Function to create sliding windows
  def createSlidingWindows(arr: Array[String], windowSize: Int, overlapSize: Int, model: Word2Vec): DataSet = {

    log.info("createSlidingWindows(): Converting words to vectors")
    val vectorArray = arr.map { word =>
      if (model.hasWord(word)) model.getWordVectorMatrix(word).toDoubleVector   // Getting vector for each word
      else Array.fill(model.getLayerSize)(0.0)  // Zero vector if word is not in the model
    }

    log.info("createSlidingWindows(): Creating sliding windows and targets")

    // Use sliding to generate windows and next elements. windowSize + 1 to get the target word
    val initialWindows = vectorArray.sliding(windowSize + 1, overlapSize).toArray

    // Check if the last part of the input was included, if not, add the last window
    val lastWindow = vectorArray.slice(vectorArray.length - windowSize - 1, vectorArray.length)

    // If the last window is not the same as the last in initialWindows, add it
    val allWindows = if (initialWindows.last.deep == lastWindow.deep) {
      initialWindows :+ lastWindow
    } else {
      initialWindows
    }

    val windowsWithTargets = allWindows.map { window =>

      // Get the main sliding window (first `windowSize` elements)
      val mainWindow = window.take(windowSize)

      // Determine the target element. If the window has more than `windowSize` elements,
      // take the last element, otherwise default to a zero vector. Just for error checking.
      val targetElement = if (window.length > windowSize) window.last else Array.fill(model.getLayerSize)(0.0)

      // Tuples of the input words with their target word
      (mainWindow, targetElement)
    }

    // Separate the sliding windows and targets into two arrays
    val slidingWindowsVectors = windowsWithTargets.map(_._1)  // Array of arrays for the windows
    val targetsVectors = windowsWithTargets.map(_._2)         // Array of words for the targets


    // Print window and target. Used for debugging
//    log.info("createSlidingWindows(): Sliding windows and corresponding targets")
//    slidingWindowsVectors.zip(targetsVectors).foreach { case (window, target) =>
//      log.info(s"Window: ${window.map(_.mkString(",")).mkString(" | ")} => Target: ${target.mkString(", ")}")
//    }

    log.info("createSlidingWindows(): Getting DataSet for windows and target")
    getTrainingData(slidingWindowsVectors, targetsVectors)
  }


   // Function to input an array of sentences (each sentence is an array of words) to get DataSet with input and output
   def getTrainingData(inputVectors: Array[Array[Array[Double]]], outputVectors: Array[Array[Double]]): DataSet = {

     // Create input INDArray and adjust the axes. This is a 3D tensor
     val inputINDArray = Nd4j.create(inputVectors).permute(0, 2, 1)

     // Create output INDArray. This is a 2D tensor
     val outputINDArray = Nd4j.create(outputVectors)

     // Return the dataset with input and outputs
     new DataSet(inputINDArray, outputINDArray)
   }


  // Define network configuration with an embedding layer
  def createMultiLayerConfiguration(): MultiLayerConfiguration = {

    log.info("createMultiLayerConfiguration(): Setting up multilayer configuration")
    val model: MultiLayerConfiguration = new NeuralNetConfiguration.Builder()
      .weightInit(WeightInit.XAVIER)
      .list()
      .layer(new LSTM.Builder()
        .nIn(3)
        .nOut(128)
        .activation(Activation.TANH)
        .build())
      .layer(new SelfAttentionLayer.Builder()
        .nIn(128)               // Input size matches the LSTM's output dimension
        .nOut(128)              // Output dimension for attention layer
        .nHeads(4)
        .projectInput(true)
        .build())
//      .layer(new DropoutLayer.Builder(0.5).build())  // Add dropout for regularization
      .layer(new DenseLayer.Builder()
        .nIn(128)
        .nOut(64)
        .activation(Activation.RELU)
        .build())
      .layer(new GlobalPoolingLayer.Builder(PoolingType.AVG).build())
      .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
        .nIn(64)
        .nOut(3)
        .activation(Activation.SOFTMAX)
        .build())
      .build()

//    val model: MultiLayerConfiguration = new NeuralNetConfiguration.Builder()
//      .weightInit(WeightInit.XAVIER)
//      .list()
//      // LSTM Layer to handle the sequence of words in each sentence
//      .layer(new LSTM.Builder()
//        .nIn(3)                // Input dimension: 10 (embedding dimension for each word)
//        .nOut(128)              // LSTM output dimension (can be tuned)
//        .activation(Activation.TANH)
//        .build())
//      // Self-Attention Layer to focus on relevant parts of the sequence
//      .layer(new SelfAttentionLayer.Builder()
//        .nIn(128)               // Input size matches the LSTM's output dimension
//        .nOut(128)              // Output dimension for attention layer
//        .nHeads(4)
//        .projectInput(true)
//        .build())
//      // Dense Layer to increase dimensionality before output
//      .layer(new DenseLayer.Builder()
//        .nIn(128)
//        .nOut(64)
//        .activation(Activation.RELU)
//        .build())
//      // Output Layer for predicting a word vector
//      .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MSE) // MSE for regression on word vectors
//        .nIn(64)
//        .nOut(3)               // Output dimension matches the word vector's dimensions
//        .activation(Activation.IDENTITY) // Identity activation for direct vector output
//        .build())
//      .build()

    model
  }

  // Method to get the number of total windows, or training sentences per DataSet
  def countWindows(n: Int, w: Int, o: Int): Int = {
    require(w > o, "Error: Window size must be greater than the overlap")
    require(n >= w, "Error: The number of words must be greater than or equal to the window size")

    // Calculate the number of full sliding windows
    val steps = (n - w).toDouble / (w - o)
    val fullWindows = Math.floor(steps).toInt + 1

    // Check if there are any remaining elements after the last full window
    val lastWindowStart = (fullWindows - 1) * (w - o) // Starting index of the last full window
    if (lastWindowStart + w < n) {
      // If the last window start plus the window size is less than total elements,
      // we need to include one more window for the remaining elements
      fullWindows + 1
    } else {
      fullWindows
    }
  }

  def main(args: Array[String]): Unit = {

    // Load the Word2Vec model from the file
    log.info("main(): Loading Word2Vec model")
    val word2Vec = WordVectorSerializer.readWord2VecModel("src/main/resources/wordvectors.txt")

    // Working with Spark
    log.info("main(): Creating SparkContext")
    val sc = createSparkContext()

    // Track the start time for metrics
    val startTime = System.currentTimeMillis()

    val word2VecBroadcast = sc.broadcast(word2Vec)

    log.info("main(): Loading input files into RDD")
    val dataFile = sc.textFile("src/main/resources/input")

    log.info("main(): Creating arrayOfWords")
    // Group words by index in blocks of 5, each block representing a sentence
    val arraysOfWords = dataFile
      .flatMap(line => line.toLowerCase().split("\\W+"))  // Split words by non-word characters
      .zipWithIndex()                       // Assign an index to each word
      .map{ case (word, index) =>
        (index / config.getInt("app.sentenceLength"), word)                   // Group words into blocks of 5
      }
      .groupByKey()                         // Group words by the calculated index
      .map{ case (groupIndex, words) =>
        words.toArray                       // Convert each group of words into an array
      }

    log.info("main(): Creating trainingData from arrayOfWords")
    val trainingData = arraysOfWords.map(sentence => createSlidingWindows(sentence,
      config.getInt("app.windowSize"), config.getInt("app.overlapSize"), word2VecBroadcast.value))

    log.info("main(): Creating MultiLayerConfiguration")
    val model: MultiLayerConfiguration = createMultiLayerConfiguration()


    log.info("main(): Creating Training Master")
    val numWindows = countWindows(config.getInt("app.sentenceLength"), config.getInt("app.windowSize"), config.getInt("app.overlapSize"))

    val trainingMaster = new ParameterAveragingTrainingMaster.Builder(numWindows)
      .batchSizePerWorker(numWindows)
      .build()

//    val trainingMaster = new SharedTrainingMaster.Builder(numWindows)
//      .batchSizePerWorker(numWindows)
//      .workersPerNode(4)  // Define number of workers per node in the cluster
//      .build()

    log.info("main(): Creating SparkDl4jMultiLayer")
    val sparkNet = new SparkDl4jMultiLayer(sc, model, trainingMaster) // Putting together spark context, neural net model
                                                                      // and training master

    sparkNet.setListeners(
      new ScoreIterationListener(10), // Log every 10 iterations
      new PerformanceListener(10)
    )
    // Train the model for numEpochs
    val numEpochs = 10
    for (epoch <- 1 to numEpochs) {
      sparkNet.fit(trainingData)
    }

    val endTime = System.currentTimeMillis()
    log.info(s"main(): Training completed in ${endTime - startTime} ms")

    log.info("main(): Stopping SparkContext")
    sc.stop()
  }
}