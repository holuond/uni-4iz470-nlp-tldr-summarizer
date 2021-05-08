package nlp_4iz470

import com.johnsnowlabs.nlp.annotator._
import com.johnsnowlabs.nlp.{DocumentAssembler, Finisher}
import nlp_4iz470.config.{AppConfig, ConfigLoader}
import org.apache.log4j.Logger
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel, IDF, IDFModel}
import org.apache.spark.sql.SparkSession

import scala.util.{Failure, Success}

object Main {
  private val logger: Logger = Logger.getLogger(getClass.getName)

  /**
   * Application's main function.
   *
   * @param args command line args
   */
  def main(args: Array[String]): Unit = {
    logger.info("Started main function")
    implicit val appConfig: AppConfig = ConfigLoader.loadAppConfig() match {
      case Success(config) => config
      case Failure(ex) => throw ex
    }

    val spark = SparkSession.builder()
      .appName("Spark NLP")
      .master("local[*]")
      .config("spark.driver.memory", "12G")
      .getOrCreate()

    // IDF Training corpus
    val trainingCorpus = List(
      (1, "GitHub is built for developer collaboration."),
      (2, "Set up a Github organization to improve the way your team works together, and get access to more features. "),
      (3, "Millions of developers and companies build, ship, and maintain their software on GitHub - the largest " +
        "and most advanced development platform in the world."))
    val trainingDF = spark.createDataFrame(trainingCorpus).toDF("id", "documentText")

    // Annotations
    val documentAssemblerAnnotation = new DocumentAssembler()
      .setInputCol("documentText")
      .setOutputCol("document")
      .setCleanupMode("shrink")

    val sentenceAnnotation = new SentenceDetector()
      .setInputCols("document")
      .setOutputCol("sentence")

    val tokenizerAnnotation = new Tokenizer()
      .setInputCols(Array("sentence"))
      .setOutputCol("token")

    val normalizerAnnotation = new Normalizer()
      .setInputCols(Array("token"))
      .setOutputCol("normalizedToken")
      .setCleanupPatterns(Array("""[^\w\d\s]"""))
      .setLowercase(true)

    val stopWordsAnnotation = new StopWordsCleaner()
      .setInputCols("normalizedToken")
      .setOutputCol("tokenNotStopWord")
      .setCaseSensitive(false)

    val lemmatizerAnnotation = LemmatizerModel
      .pretrained
      .setInputCols(Array("tokenNotStopWord"))
      .setOutputCol("lemma")

    val lemmaArrayAnnotation = new Finisher()
      .setInputCols("lemma")
      .setOutputCols("lemmaArray")
      .setCleanAnnotations(true) // Whether to remove intermediate annotations
      .setOutputAsArray(true) // Whether to output as Array. Useful as input for other Spark transformers.

    val countVectorizerAnnotation = new CountVectorizer()
      .setInputCol("lemmaArray")
      .setOutputCol("features")

    val idf = new IDF()
      .setInputCol("features")
      .setOutputCol("idf")

    // Pipeline
    val pipeline = new Pipeline()
      .setStages(Array(
        documentAssemblerAnnotation,
        sentenceAnnotation,
        tokenizerAnnotation,
        normalizerAnnotation,
        stopWordsAnnotation,
        lemmatizerAnnotation,
        lemmaArrayAnnotation,
        countVectorizerAnnotation,
        idf
      ))

    // Train model
    val model = pipeline.fit(trainingDF)

    val idfModel: IDFModel = model.stages.last.asInstanceOf[IDFModel]
    val vectorizerModel: CountVectorizerModel = model.stages.apply(pipeline.getStages.length - 2).asInstanceOf[CountVectorizerModel]

    // Print vocabulary and IDF vector
    println(vectorizerModel.vocabulary.zip(idfModel.idf.toArray).mkString(" "))

    // Test data
    val testCorpus = List((1, "GitHub organization software"))
    val testDF = spark.createDataFrame(testCorpus).toDF("id", "documentText")

    // Score test data with IDF model
    val pipelineDF = pipeline.fit(trainingDF).transform(testDF)

    // Print results
    pipelineDF.show(100, truncate = false)

    // Annotation.collect(pipelineDF, "idf").flatten.toSeq.foreach(println)
  }
}
