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
import org.apache.hadoop.fs.Path
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.util.ModelSerializer

import java.io.PrintWriter

object SlidingWindowTraining {

  private val log = LoggerFactory.getLogger("SlidingWindowTraining")
  private val config = ConfigFactory.load()

  /**
   * This function is used to initialize the SparkContext with its configuration. You can set the master
   * for local, yarn, mesos, etc.
   * @return  The SparkContext to run Spark
   */
  def createSparkContext(master: String): SparkContext = {
    // Configure Spark for local or cluster mode
    log.info("createSparkContext(): Setting up SparkConf")
    val conf = new SparkConf()
      .setAppName("Sliding Window Training")
      .setMaster(master) // For local testing, or use "yarn", "mesos", etc. in a cluster

    new SparkContext(conf)
  }

  /**
   * This function is used to create sliding windows from a group of ordered words (a sentence, or sentences). It takes
   * the text, retrieves the vector embedding for each word, computes the windows of windowSize, with overlapSize overlap
   * for each window, and calls a helper function to transform the input, a 3D tensor representation of
   * [number of sentences, number of words per sentence, number of dimensions for each vector] and the target labels,
   * a 2D tensor with [target word for each sentence, number of dimensions] that holds the word following each sentence
   * into a DataSet object used for training the neural network model.
   * @param arr   A sequence of ordered words or text
   * @param windowSize  The window size for the sliding windows
   * @param overlapSize    The step size for how many elements to advance in each window
   * @param model   The Word2Vec model containing the vector embeddings for each word
   * @return  A DataSet object with input and output features
   */
  def createSlidingWindows(arr: Array[String], windowSize: Int, overlapSize: Int, model: Word2Vec, embeddingSize: Int): DataSet = {

    log.info("createSlidingWindows(): Converting words to vectors")
    val vectorArray = arr.map { word =>
      if (model.hasWord(word)) model.getWordVectorMatrix(word).toDoubleVector
      else Array.fill(embeddingSize)(0.0)
    }

    // If vectorArray is shorter than required, pad it upfront
    val paddedVectorArray = if (vectorArray.length < windowSize + 1) {
      vectorArray.padTo(windowSize + 1, Array.fill(embeddingSize)(0.0))
    } else {
      vectorArray
    }

    log.info("createSlidingWindows(): Creating sliding windows and targets")

    // Use sliding to generate windows
    val initialWindows = paddedVectorArray.sliding(windowSize + 1, windowSize - overlapSize).toArray

    // Get the last window with all the elements
    val lastWindow = paddedVectorArray.slice(vectorArray.length - windowSize - 1, vectorArray.length)

    // if last window is incomplete and has fewer elements, replace it with another
    val allWindows = if (!initialWindows.last.sameElements(lastWindow)) {
      initialWindows.filterNot(_.sameElements(initialWindows.last)) :+ lastWindow
    } else {
      initialWindows
    }

    // Separate the windows from their targets
    val windowsWithTargets = allWindows.map { window =>
      val mainWindow = window.take(windowSize)
      val targetElement = window.last
      (mainWindow, targetElement)
    }

    val slidingWindowsVectors = windowsWithTargets.map(_._1)  // The sliding windows
    val targetsVectors = windowsWithTargets.map(_._2)   // The target for each window

    log.info("createSlidingWindows(): Getting DataSet for windows and target")
    getTrainingData(slidingWindowsVectors, targetsVectors)
  }

  /**
   * Function to produce the training data used in the neural network model
   * @param inputVectors  A 3D tensor with shape [number of sentences, number of words per sentence,
   *                      number of dimensions for each vector]. These are the training sentences
   * @param outputVectors   2D tensor with [target word for each sentence, number of dimensions]. These are the
   *                        target words for training
   * @return  A DataSet object containing the input sentences with their corresponding target words
   */
  def getTrainingData(inputVectors: Array[Array[Array[Double]]], outputVectors: Array[Array[Double]]): DataSet = {

    // Create input INDArray and adjust the axes
    val inputINDArray = Nd4j.create(inputVectors).permute(0, 2, 1)
    val outputINDArray = Nd4j.create(outputVectors)

    new DataSet(inputINDArray, outputINDArray)
  }

  /**
   * This function creates the MultiLayerConfiguration object used for the neural network training.
   * It sets the layers for the configuration and builds it.
   * @param embeddingSize The embedding dimensions of the words used in the input
   * @return  A MultiLayerConfiguration used for training
   */
  def createMultiLayerConfiguration(embeddingSize: Int): MultiLayerConfiguration = {

    log.info("createMultiLayerConfiguration(): Setting up multilayer configuration")
    val model: MultiLayerConfiguration = new NeuralNetConfiguration.Builder()
      .weightInit(WeightInit.XAVIER)
      .list()
      .layer(new LSTM.Builder()
        .nIn(embeddingSize)  // The embedding dimensions of the vectors
        .nOut(128)
        .activation(Activation.TANH)
        .build())
      .layer(new SelfAttentionLayer.Builder()
        .nIn(128)
        .nOut(128)
        .nHeads(4)
        .projectInput(true)
        .build())
      .layer(new DenseLayer.Builder()
        .nIn(128)
        .nOut(64)
        .activation(Activation.RELU)
        .build())
      .layer(new GlobalPoolingLayer.Builder(PoolingType.AVG).build())
      .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
        .nIn(64)
        .nOut(embeddingSize)   // The embedding dimensions of the vectors
        .activation(Activation.IDENTITY)
        .build())
      .build()

    model
  }

  /**
   * Function to calculate the number of total windows given by a sequence of words (text)
   * @param numWords  The number of words per sentence
   * @param windowSize  The window size for sliding windows
   * @param overlapSize   The overlap size for the windows
   * @return  The total number of windows
   */
  def countWindows(numWords: Int, windowSize: Int, overlapSize: Int): Int = {
    require(windowSize > overlapSize, "Error: Window size must be greater than the overlap")
    require(numWords >= windowSize, "Error: The number of words must be greater than or equal to the window size")

    // Calculate the number of full sliding windows
    val steps = (numWords - windowSize).toDouble / (windowSize - overlapSize)
    val fullWindows = Math.floor(steps).toInt + 1

    // Check if there are any remaining elements after the last full window
    val lastWindowStart = (fullWindows - 1) * (windowSize - overlapSize) // Starting index of the last full window
    if (lastWindowStart + windowSize < numWords) {
      fullWindows + 1
    } else {
      fullWindows
    }
  }

  /**
   * This is the main function and driver for the program. It loads a Word2Vec model with vector embeddings, reads a
   * directory of .txt files for input, splits the text and calls functions to get sliding windows of the text with
   * their targets as tensor representations RDDs, initializes a NN for training on sequence prediction, trains the NN
   * using the sliding windows using distributed training, and collects metrics of performance of the program.
   * It takes arguments for the path of the wWrd2Vec model, the input directory, the path to write the NN model, and the
   * path to write the stats file.
   * @param args  Paths for the existing Word2Vec model, the input directory, the path to write the NN model, and the
   *              * path to write the stats file.
   */
  def main(args: Array[String]): Unit = {

    // Load the Word2Vec model from the file
    log.info("main(): Loading Word2Vec model")

    val word2VecPath = new Path(args(0)).toString // "s3://jonathan-homework2/input/word-vectors-medium.txt" // "src/main/resources/word-vectors.txt"
    val inputDir = new Path(args(1)).toString // "s3://jonathan-homework2/input/input-small/"  // "src/main/resources/input"
    val modelFile = new Path(args(2)).toString // "s3://jonathan-homework2/output/model.zip"   // "src/main/resources/model.zip"
    val statsFile = new Path(args(3)).toString // "s3://jonathan-homework2/output/training-stats.txt"   // "src/main/resources/training-stats.txt"

    val word2Vec = WordVectorSerializer.readWord2VecModel(word2VecPath)

    // Setting up file to write stats
    val writer = new PrintWriter(statsFile)

    // Working with Spark
    log.info("main(): Creating SparkContext")
    val sc = createSparkContext("local[*]")   // Set master for SparkContext

    // Track the start time for metrics
    val startTime = System.currentTimeMillis()

    val word2VecBroadcast = sc.broadcast(word2Vec)

    log.info("main(): Loading input files into RDD")
    val dataFile = sc.textFile(inputDir)

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
      config.getInt("app.windowSize"), config.getInt("app.overlapSize"), word2VecBroadcast.value, config.getInt("app.embeddingDimensions")))

    val numPartitions = trainingData.getNumPartitions
    log.info(s"Number of partitions: $numPartitions")
    writer.println(s"Number of partitions: $numPartitions")

    log.info("main(): Creating MultiLayerConfiguration")
    val model: MultiLayerConfiguration = createMultiLayerConfiguration(config.getInt("app.embeddingDimensions"))

    log.info("main(): Creating Training Master")
    val numWindows = countWindows(config.getInt("app.sentenceLength"), config.getInt("app.windowSize"),
      config.getInt("app.overlapSize"))

    val trainingMaster = new ParameterAveragingTrainingMaster.Builder(numWindows)
      .batchSizePerWorker(numWindows)
      .averagingFrequency(5)   // Frequency of parameter averaging
      .workerPrefetchNumBatches(2)
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
    (1 to numEpochs).foreach { epoch =>
      val epochStart = System.currentTimeMillis()

      sparkNet.fit(trainingData)

      val epochEnd = System.currentTimeMillis()
      val epochTime = epochEnd - epochStart

      log.info(s"Epoch $epoch time: ${epochTime}ms")

      // Write the epoch time to the file
      writer.println(s"Epoch $epoch time: ${epochTime}ms")
      writer.println(s"Epoch $epoch learning rate: ${sparkNet.getNetwork.getLearningRate(0)}")
    }

    // Save the trained model
    log.info(s"main(): Saving model to $modelFile")
    ModelSerializer.writeModel(sparkNet.getNetwork, modelFile, true)

    val endTime = System.currentTimeMillis()

    val totalTime = endTime - startTime

    log.info(s"main(): Program completed in $totalTime ms")

    writer.println(s"Program completed in: $totalTime ms")

    // Close the file writer after all epochs
    writer.close()

    log.info("main(): Stopping SparkContext")
    sc.stop()
  }
}