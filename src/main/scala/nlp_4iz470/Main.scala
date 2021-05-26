package nlp_4iz470

import nlp_4iz470.config.{AppConfig, ConfigLoader}
import org.apache.log4j.Logger
import org.apache.spark.ml.PipelineModel
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
      .appName("TLDR Summarizer")
      .master("local[*]")
      .config("spark.driver.memory", "12G")
      .getOrCreate()

    // Train or load pre-trained IDF model
    val idfPipelineModel: PipelineModel =
      IdfPipeline.loadOrTrain(
          appConfig.idfModelPath,
          Some(appConfig.trainingCorpusDir),
          Some(appConfig.modelsDir))

    // Test data
    val testCorpus = List((1, "GitHub is a tool for developers."))
    val testDF = spark.createDataFrame(testCorpus).toDF("id", "documentText")

    // Score test data with IDF model
    val pipelineDF = idfPipelineModel.transform(testDF)

    // Print results
    pipelineDF.show(1000, truncate = false)
  }
}
