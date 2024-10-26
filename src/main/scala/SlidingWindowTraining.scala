import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer
import org.deeplearning4j.nn.conf.MultiLayerConfiguration
import org.nd4j.linalg.dataset.DataSet
import org.apache.spark.rdd.RDD

import org.deeplearning4j.nn.conf.layers.{GlobalPoolingLayer, LSTM, OutputLayer, PoolingType}
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


object SlidingWindowTraining {

  private val log = LoggerFactory.getLogger("SlidingWindowTraining")
  private val config = ConfigFactory.load()

  private def createSparkContext(): SparkContext = {
    // Configure Spark for local or cluster mode
    log.info("createSparkContext(): Setting up SparkConf")
    val conf = new SparkConf()
      .setAppName("Sample App")
      .setMaster("local[*]") // For local testing, or use "yarn", "mesos", etc. in a cluster

    new SparkContext(conf)
  }

  // Function to create sliding windows
  private def createSlidingWindows(arr: Array[String], windowSize: Int, overlapSize: Int, model: Word2Vec): DataSet = {

    log.info("createSlidingWindows(): Converting words to vectors")
    val vectorArray = arr.map { word =>
      if (model.hasWord(word)) model.getWordVectorMatrix(word).toDoubleVector   // Getting vector for each word
      else Array.fill(model.getLayerSize)(0.0)  // Zero vector if word is not in the model
    }

    // Use sliding to generate windows and next elements. windowSize + 1 to get the target word
    log.info("createSlidingWindows(): Creating sliding windows and targets")
    val windowsWithTargets = vectorArray.sliding(windowSize + 1, overlapSize).toArray.map { window =>
      val mainWindow = window.take(windowSize)  // Take the main sliding window
      val targetElement = window.lastOption.getOrElse(Array.fill(model.getLayerSize)(0.0))  // Default to zero vector if none exists
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
   private def getTrainingData(inputVectors: Array[Array[Array[Double]]], outputVectors: Array[Array[Double]]): DataSet = {

     // Create input INDArray and adjust the axes. This is a 3D tensor
     val inputINDArray = Nd4j.create(inputVectors).permute(0, 2, 1)

     // Create output INDArray. This is a 2D tensor
     val outputINDArray = Nd4j.create(outputVectors)

     // Return the dataset with input and outputs
     new DataSet(inputINDArray, outputINDArray)
   }


  // Define network configuration with an embedding layer
  private def createMultiLayerConfiguration(): MultiLayerConfiguration = {

    log.info("createMultiLayerConfiguration(): Setting up multilayer configuration")
    val model: MultiLayerConfiguration = new NeuralNetConfiguration.Builder()
      .list()
      .layer(new LSTM.Builder()
        .nIn(config.getInt("app.embeddingDimensions"))
        .nOut(config.getInt("app.embeddingDimensions"))
        .activation(Activation.TANH)
        .build())
      .layer(new GlobalPoolingLayer.Builder(PoolingType.AVG).build())
      .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
        .nIn(config.getInt("app.embeddingDimensions"))
        .nOut(config.getInt("app.embeddingDimensions"))
        .activation(Activation.SOFTMAX)
        .build())
      .build()

    model
  }

  // Method to get the number of total windows, or training sentences per DataSet
  def countWindows(n: Int, w: Int, o: Int): Int = {
    require(w > o, "Error: Window size must be greater than the overlap")
    require(n >= w, "Error: The number of words must be greater than or equal to the window size")

    // Apply the formula: (n - w) / (w - o) + 1
    val steps = (n - w).toDouble / (w - o)
    Math.floor(steps).toInt + 1
  }

  def main(args: Array[String]): Unit = {

    // Load the Word2Vec model from the file
    log.info("main(): Loading Word2Vec model")
    val word2Vec = WordVectorSerializer.readWord2VecModel("src/main/resources/wordvectors.txt")

    // Working with Spark
    log.info("main(): Creating SparkContext")
    val sc = createSparkContext()

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

    log.info("main(): Creating ParameterAveragingTrainingMaster")
    val trainingMaster = new ParameterAveragingTrainingMaster.Builder(countWindows(config.getInt("app.sentenceLength"),
      config.getInt("app.windowSize"), config.getInt("app.overlapSize"))).build()

    log.info("main(): Creating SparkDl4jMultiLayer")
    val sparkNet = new SparkDl4jMultiLayer(sc, model, trainingMaster) // Putting together spark context, neural net model
                                                                      // and training master
    // Train the model for numEpochs
    val numEpochs = 10
    for (epoch <- 1 to numEpochs) {
      sparkNet.fit(trainingData)
    }

    log.info("main(): Stopping SparkContext")
    sc.stop()
  }
}