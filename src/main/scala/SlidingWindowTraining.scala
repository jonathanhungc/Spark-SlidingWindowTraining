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
  private def createSlidingWindows(arr: Array[String], windowSize: Int, model: Word2Vec): DataSet = {

    // Use sliding to generate windows and next elements
    log.info("createSlidingWindows(): Creating sliding windows and targets")
    val windowsWithTargets = arr.sliding(windowSize + 1, 1).toArray.map { window =>
      val mainWindow = window.take(windowSize)  // Take the main sliding window
      val nextElement = window.lastOption.getOrElse("")  // Get the next element, or empty if none exists
      (mainWindow, nextElement)
    }

    // Separate the sliding windows and targets into two arrays
    val slidingWindows = windowsWithTargets.map(_._1)  // Array of arrays for the windows
    val targets = windowsWithTargets.map(_._2)         // Array of words for the targets


    log.info("createSlidingWindows(): Sliding windows and corresponding targets")
    slidingWindows.zip(targets).foreach { case (window, target) =>
      log.info(s"Window: ${window.mkString(", ")} => Target: $target")
    }

    log.info("createSlidingWindows(): Getting DataSet for windows and target")
    getTrainingData(slidingWindows, targets, model)

  }


   // Function to input an array of sentences (each sentence is an array of words) to get DataSet with input and output
   private def getTrainingData(inputSentences: Array[Array[String]], outputWords: Array[String], model: Word2Vec): DataSet = {

     // Helper function to get the word vector or a zero vector if the word is not found
     def getWordVector(word: String): Array[Double] = {
       if (model.hasWord(word)) {
         model.getWordVectorMatrix(word).toDoubleVector
       } else {
         log.info(s"getTrainingData(): Error word: $word")
         Array.fill(model.getLayerSize)(0.0)
       }
     }

     // Map input sentences to their word vectors
     val inputTensor = inputSentences.map(sentence => sentence.map(getWordVector))

     // Create input INDArray and adjust the axes. This is a 3D tensor
     val inputINDArray = Nd4j.create(inputTensor).permute(0, 2, 1)

     // Map output words to their word vectors
     val outputLabels = outputWords.map(getWordVector)

     // Create output INDArray. This is a 2D tensor
     val outputINDArray = Nd4j.create(outputLabels)

     log.info(s"Input array: $inputINDArray")
     log.info(s"Output array: $outputINDArray")

     // Return the dataset with input and outputs
     new DataSet(inputINDArray, outputINDArray)
   }


  // Define network configuration with an embedding layer
  private def createMultiLayerConfiguration(): MultiLayerConfiguration = {

    log.info("createMultiLayerConfiguration(): Setting up multilayer configuration")
    val model: MultiLayerConfiguration = new NeuralNetConfiguration.Builder()
      .list()
      .layer(new LSTM.Builder()
        .nIn(3)
        .nOut(3)
        .activation(Activation.TANH)
        .build())
      .layer(new GlobalPoolingLayer.Builder(PoolingType.AVG).build())
      .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
        .nIn(3)
        .nOut(3)
        .activation(Activation.SOFTMAX)
        .build())
      .build()

    model
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
      .flatMap(line => line.split("\\W+"))  // Split words by non-word characters
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
      config.getInt("app.windowSize"), word2VecBroadcast.value))

    log.info("main(): Creating MultiLayerConfiguration")
    val model: MultiLayerConfiguration = createMultiLayerConfiguration()

    log.info("main(): Creating ParameterAveragingTrainingMaster")
    val trainingMaster = new ParameterAveragingTrainingMaster.Builder(config.getInt("app.examplesPerDataSetObject")).build()

    log.info("main(): Creating SparkDl4jMultiLayer")
    val sparkNet = new SparkDl4jMultiLayer(sc, model, trainingMaster) // Putting together spark context, neural net model
                                                                      // and training master
    // Train the model for numEpochs
    val numEpochs = 10
    for (epoch <- 1 to numEpochs) {
      sparkNet.fit(trainingData)
    }

    println("main(): Stopping SparkContext")
    sc.stop()
}
}