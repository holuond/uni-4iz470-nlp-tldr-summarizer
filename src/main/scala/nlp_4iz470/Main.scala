package nlp_4iz470

import nlp_4iz470.config.{AppConfig, ConfigLoader}
import org.apache.commons.io.FileUtils
import org.apache.log4j.Logger
import org.apache.spark.ml.PipelineModel
import org.apache.spark.sql.SparkSession

import java.io.File
import java.nio.file.Path
import scala.io.Source
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

    logger.info("Loading application configuration")
    implicit val appConfig: AppConfig = ConfigLoader.loadAppConfig() match {
      case Success(config) => config
      case Failure(ex) => throw ex
    }

    // Initialize a Spark session
    SparkSession.builder()
      .appName("TLDR Summarizer")
      .master("local[*]")
      .config("spark.driver.memory", "12G")
      .getOrCreate()

    // Load a pre-trained IDF model if possible, otherwise train a new model
    val idfPipelineModel: PipelineModel =
      IdfPipeline.loadOrTrain(
        appConfig.idfModelPath,
        Some(appConfig.trainingCorpusDir),
        Some(appConfig.modelsDir))

    // Run summarization for each txt document found in the input directory
    val inputDocuments: Seq[Path] = locateInputDocuments(appConfig.inputDocumentsDir)
    inputDocuments.foreach {
      docPath =>
        logger.info(s"************* Starting execution for input file: ${docPath.getFileName.toString} *************")

        // Extract text
        val source = Source.fromFile(docPath.toString)
        val documentText = source.getLines.mkString(" ")
        source.close

        // Summarize
        Summarizer.summarizeDocument(documentText, idfPipelineModel)

      /*
      // Compare results
      logger.info("*********** Compare results to JohnSnowLabs T5 Transformer ***********")
      Summarizer.summarizeWithT5(documentText)
      */
    }
  }

  /**
   * Return seq of paths pointing to files with given extensions in the given directory
   *
   * @param inputDir   dir to look in (non-recursive)
   * @param extensions file extensions to include
   * @return seq of paths pointing to files with given extensions in the given directory
   */
  def locateInputDocuments(inputDir: Path, extensions: Array[String] = Array("txt")): Seq[Path] = {
    FileUtils.listFiles(new File(inputDir.toString), extensions, false)
      .toArray.toList.asInstanceOf[List[File]].map(_.toPath)
  }
}
