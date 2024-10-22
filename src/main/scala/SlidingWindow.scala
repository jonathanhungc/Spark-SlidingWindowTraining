import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.rdd.RDD
import org.deeplearning4j.nn.conf.MultiLayerConfiguration
import org.deeplearning4j.nn.conf.layers.{GlobalPoolingLayer, LSTM, OutputLayer, PoolingType}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.optimize.api.IterationListener
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.spark.api.TrainingMaster
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster
import org.nd4j.linalg.lossfunctions.LossFunctions
//import org.deeplearning4j.spark.parameterserver.training.SharedTrainingMaster
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.apache.spark.api.java.JavaRDD

import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.api.ndarray.INDArray
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.EmbeddingLayer
import org.nd4j.linalg.activations.Activation

import org.deeplearning4j.models.word2vec.Word2Vec
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer

object SlidingWindow {

  private def createSparkContext(): SparkContext = {
    // Configure Spark for local or cluster mode
    val conf = new SparkConf()
      .setAppName("Sample App")
      .setMaster("local[*]") // For local testing, or use "yarn", "mesos", etc. in a cluster

    new SparkContext(conf)
  }

  // Function to create sliding windows
  def createSlidingWindows(arr: Array[Array[String]], windowSize: Int): Array[DataSet] = {
    arr.sliding(windowSize + 1).map(window => {
      val mainWindow = window.take(windowSize)        // Take the main sliding window
      val nextElement = window.lift(windowSize)       // Get the next element (if it exists)
      (mainWindow, nextElement)                       // Return the window and the next element
    })

  }

  // Function to input an array of sentences (each sentence is an array of words) to get DataSet with input and output
  def getTrainingData(inputSentences: Array[Array[String]], outputWords: Array[String], model: Word2Vec): DataSet = {

    // For each sentence, take each word in the sentence and get its vector
    val inputTensor = inputSentences.map(sentence => sentence.map(word =>
      model.getWordVectorMatrix(word).toDoubleVector))

    val inputINDArray = Nd4j.create(inputTensor).permute(0, 2, 1)

    val outputLabels = outputWords.map(word => {
      model.getWordVectorMatrix(word).toDoubleVector
    })

    val outputINDArray = Nd4j.create(outputLabels)

    new DataSet(inputINDArray, outputINDArray)
  }


  def main(args: Array[String]): Unit = {

    // Load the Word2Vec model from the file
    val word2Vec = WordVectorSerializer.readWord2VecModel("src/main/resources/wordvectors.txt")

    System.out.println(word2Vec.hasWord("word1"))

    System.out.println(word2Vec.getWordVectorMatrix("word1"))

    // Working with Spark
    val sc = createSparkContext()

    val dataFile = sc.textFile("src/main/resources/input")  // get file with data

    val indexedWordsRDD = dataFile.flatMap(line => line.split("\\W+")).zipWithIndex()

    // Group words by their index in blocks of `wordsPerGroup`
    val groupedWordsRDD = indexedWordsRDD.map{ case (word, index) =>
      (index / 5, word)  // Assign each word to a group
    }.groupByKey()  // Group words by the calculated index

    // Convert to arrays of words. Each array corresponds to a whole continuous sentences.
    // arrayOfWords is an RDD[Array[String]]
    val arraysOfWords = groupedWordsRDD.map{ case (groupIndex, words) =>
      words.toArray  // Convert each group into an array of words
    }

    val indexedArraysRDD = arraysOfWords.zipWithIndex()

    // Group arrays of words into groups of 5
    val groupedArraysRDD = indexedArraysRDD.map{ case (array, index) =>
      (index / 5, array)  // Group arrays by dividing the index by 5
    }.groupByKey()

    // Convert grouped arrays into a list of 5 arrays (sentences) per group
    val groupedSentencesRDD = groupedArraysRDD.map { case (groupIndex, arrays) =>
      arrays.toArray  // Convert the grouped arrays to an array (containing 5 arrays)
    }

    val trainingData = groupedSentencesRDD.flatMap(sentence => createSlidingWindows(sentence, 3))

    println("stopping sparkContext")
    sc.stop()

    // Define network configuration with an embedding layer
//    val config: MultiLayerConfiguration = new NeuralNetConfiguration.Builder()
//      .list()
//      .layer(new LSTM.Builder()
//        .nIn(3)
//        .nOut(3)
//        .activation(Activation.TANH)
//        .build())
//      .layer(new GlobalPoolingLayer.Builder(PoolingType.AVG).build())
//      .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
//        .nIn(3)
//        .nOut(3)
//        .activation(Activation.SOFTMAX)
//        .build())
//      .build()
//
//    // Initialize the model with the defined configuration
//    val model = new MultiLayerNetwork(config)
//    model.init()
//
//    val inputData: Array[Array[Array[Double]]] = Array(
//      Array( // First sentence
//        Array(0.1, 0.2, 0.3), // Word 1
//        Array(0.4, 0.5, 0.6), // Word 2
//        Array(0.7, 0.8, 0.9),  // Word 3
//        Array(1.0, 1.1, 1.2)
//      ),
//      Array( // Second sentence
//        Array(1.0, 1.1, 1.2), // Word 1
//        Array(1.3, 1.4, 1.5), // Word 2
//        Array(1.6, 1.7, 1.8),  // Word 3
//        Array(1.9, 2.0, 2.1)
//      )
//    )
//
//    // Create an INDArray for the input
//    val inputFeatures: INDArray = Nd4j.create(inputData).permute(0, 2, 1)
//
//    // Output: Target word (represented as a 3-dimensional vector) for each sentence
//    val outputData: Array[Array[Double]] = Array(
//      Array(0.9, 0.8, 0.7),  // Target for first sentence
//      Array(1.8, 1.7, 1.6)   // Target for second sentence
//    )
//
//    // Create an INDArray for the output
//    val outputLabels: INDArray = Nd4j.create(outputData)
//
////    // Print both input and output to verify
//    println("Input INDArray:")
//    println(inputFeatures)
//
//    println("\nOutput INDArray:")
//    println(outputLabels)
//
//    println(s"Input shape: ${inputFeatures.shape.mkString(", ")}") // Should output: 2, 3, 4
//    println(s"Output shape: ${outputLabels.shape.mkString(", ")}") // Should output: 2, 3
//
//
//    model.fit(new DataSet(inputFeatures, outputLabels))
  }
}