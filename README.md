# Homework 2: Spark Sliding Window Training
### Name: Jonathan Hung

#### UIC email: jhung9@uic.edu

This is a program to train a neural network for text sequence prediction using Spark and its distributed computing
capabilities. In the /resources directory, you can find different input text files that I used for testing and
training the model locally, as well as files that contain the vector embeddings for all the words present in the
input files. So, the important files and directories are:

- **src/main/scala/SlidingWindowTraining.scala**: This file contains all the logic for the program. This file holds the
code that reads a directory of .txt files, split all the text into sentences of a fixed number of words, computes
sliding windows with their corresponding targets over each sentence and stores them in RDDs, takes the windows and their
targets and feeds them as vector representations (a 3D tensor and a 2D tensor, respectively) to a neural network for
sequence prediction, and finally saves the model.
- **src/main/resources**: Holds all the input data for the program. The directory /input has a few .txt files, small
amount of data, and its corresponding file with vector embeddings is word-vectors.txt. Likewise, /input-large has 2 .txt
files (around 4MB) and its corresponding vector embedding file is word-vectors-medium.txt. The input directories hold
training text, while the word vectors files hold vector embeddings.
- **src/test/scala/Test.scala**: This file has all the tests for the functions used in the main file
(SlidingWindowTraining.scala). It tests that the functions used for splitting and transforming the data work correctly,
and that they provide the expected format for training. It also tests that the SparkContext and neural network are
created correctly. For the scala tests to run correctly, the specified variables 
- **build.sbt**: This file has all the dependencies of the program, including Apache Spark, DeepLearning4j, etc.
- **src/main/resources/application.conf**: This file sets different variables in the program, according  to the input data.
There's one configuration for the Scala tests, and one for the actual extensive testing.
- **Input**: A directory with .txt files with sentences, and a .txt file with the vector embeddings of each word
- **Output**: A NN model saved by the program after training, and a training-stats.txt file with the stats of the training
- The input and output paths must be given when executing the program, and the variables in application.conf must be set
according to the input. For instance, in the below running commands the program will run with around 4MB of data. The
input directory in this case is src/main/resources/input-medium, and the word vector embeddings file for this directory 
is src/main/resources/word-vectors-medium.txt. After execution, the NN model will be saved to src/main/resources/model.zip
and a file with the stats of training will be written to src/main/resources/training-stats.txt.
### To run the program execute the following commands (you may want to delete the src/main/resources/output directory before running this):
```
sbt clean
sbt compile
sbt "run src/main/resources/word-vectors-medium.txt src/main/resources/input-medium src/main/resources/model.zip src/main/resources/training-stats.txt"

```

### The video of the deployment on Amazon EMR (Elastic Map Reduce) is found [here](https://youtu.be/qI8PZPiBnFM)