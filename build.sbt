ThisBuild / version := "0.1.0-SNAPSHOT"

ThisBuild / scalaVersion := "2.12.15"

//assembly / assemblyMergeStrategy := {
//  case PathList("META-INF", xs @ _*) =>
//    xs match {
//      case "MANIFEST.FM" :: Nil => MergeStrategy.discard
//      case "services" :: _      => MergeStrategy.concat
//      case _                    => MergeStrategy.discard
//    }
//  case "reference.conf" => MergeStrategy.concat
//  case x if x.endsWith(".proto") => MergeStrategy.rename
//  case x if x.contains("hadoop") => MergeStrategy.first
//  case _ => MergeStrategy.first
//}
//
//lazy val root = (project in file("."))
//  .settings(
//    name := "homework2",
//    assembly / mainClass := Some("SlidingWindow"),
//    assembly / assemblyJarName := "map-reduce-word2vec.jar",
//
//    libraryDependencies += "org.apache.hadoop" % "hadoop-common" % "3.4.0",
//    libraryDependencies += "org.apache.hadoop" % "hadoop-mapreduce-client-core" % "3.4.0",
//    libraryDependencies += "org.apache.hadoop" % "hadoop-mapreduce-client-jobclient" % "3.4.0",
//
//    libraryDependencies += "com.knuddels" % "jtokkit" % "1.1.0",
//
//    libraryDependencies += "org.slf4j" % "slf4j-api" % "2.0.16",
//
//    libraryDependencies += "ch.qos.logback" % "logback-classic" % "1.5.6",
//
//    libraryDependencies += "com.typesafe" % "config" % "1.4.3",
//
//    libraryDependencies += "org.deeplearning4j" % "deeplearning4j-core" % "1.0.0-M2.1",
//
//    libraryDependencies += "org.deeplearning4j" % "deeplearning4j-ui-model" % "1.0.0-M2.1",
//
//    libraryDependencies += "org.deeplearning4j" % "deeplearning4j-nlp" % "1.0.0-M2.1",
//
//    libraryDependencies += "org.nd4j" % "nd4j-native-platform" % "1.0.0-M2.1",
//
//    libraryDependencies += "org.scalatest" %% "scalatest" % "3.2.19" % "test",
//
//    libraryDependencies += "org.mockito" % "mockito-core" % "3.12.4" % "test",
//
//    libraryDependencies += "org.deeplearning4j" % "deeplearning4j-spark" % "1.0.0-M2.1",
//
//    libraryDependencies += "org.apache.spark" %% "spark-core" % "3.5.1",
//
//    libraryDependencies += "org.apache.spark" %% "spark-mllib" % "3.5.1"
//  )

libraryDependencies ++= Seq(
//  "org.apache.spark" %% "spark-core" % "3.4.1",
//  "org.apache.spark" %% "spark-sql"  % "3.sb4.1",
//  "org.scalatest" %% "scalatest" % "3.2.2" % Test, // For unit testing

  "org.apache.spark" %% "spark-core" % "3.5.1",

  "org.apache.spark" %% "spark-mllib" % "3.5.1",

  "org.deeplearning4j" % "deeplearning4j-core" % "1.0.0-M2.1",
  "org.deeplearning4j" % "deeplearning4j-ui-model" % "1.0.0-M2.1",
  "org.deeplearning4j" % "deeplearning4j-nlp" % "1.0.0-M2.1",
  "org.nd4j" % "nd4j-native-platform" % "1.0.0-M2.1",

  "org.deeplearning4j" % "dl4j-spark_2.12" % "1.0.0-M2.1",

  "org.scalatest" %% "scalatest" % "3.2.19" % "test",

)

ThisBuild / fork := true  // Forks a JVM to avoid heap space issues
