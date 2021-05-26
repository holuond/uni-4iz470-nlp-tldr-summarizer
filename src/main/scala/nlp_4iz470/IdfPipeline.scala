package nlp_4iz470

import com.johnsnowlabs.nlp.annotator._
import com.johnsnowlabs.nlp.{DocumentAssembler, Finisher}
import nlp_4iz470.ImplicitConversions._
import org.apache.commons.io.FilenameUtils
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.log4j.Logger
import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel, IDF, IDFModel}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.{col, length, monotonically_increasing_id}

import java.time.LocalDateTime

object IdfPipeline {
  private val logger: Logger = Logger.getLogger(getClass.getName)

  /**
   * If idfModelPath is set, returns a pretrained model by loading it from the path,
   * otherwise trains a new model based on trainingCorpusPath (one of these Options has to be defined).
   * If a new model is trained it is saved to disk to modelsDir, if it is defined.
   *
   * @param idfModelPath path to a pre-trained IDF model
   * @param trainingCorpusDir path to a directory that contains the training corpus
   * @param modelsDir path to a directory where trained models are saved
   * @return a trained [[IDFModel]]
   */
  def loadOrTrain(idfModelPath: Option[Path], trainingCorpusDir: Option[Path], modelsDir: Option[Path]): PipelineModel = {
    (idfModelPath, trainingCorpusDir, modelsDir) match {
      case (Some(idfModelPath), _, _) =>
        // Load
        logger.info(s"Loading pretrained IDF model from ${idfModelPath}")
        PipelineModel.load(idfModelPath.toString)

      case (None, Some(trainingCorpusDir), modelsDir) =>
        // Train
        logger.info("No pretrained IDF model specified, continuing to training.")
        val corpus: Seq[Path] = getCorpusFiles(trainingCorpusDir, Seq("txt"))
        logger.info(s"${corpus.length} input files will be used as the IDF training corpus:\n" + corpus.mkString("\n"))
        train(corpus, modelsDir)

      case x => throw new IllegalArgumentException(s"loadOrTrainModel cannot handle parameters $x")
    }
  }

  /**
   * Return seq of paths pointing to files that form the model corpus
   *
   * @param corpusDir  dir to look in (non-recursive)
   * @param extensions file extensions to include
   * @return seq of paths pointing to files that form the model corpus
   */
  private def getCorpusFiles(corpusDir: Path, extensions: Seq[String]): Seq[Path] = {
    val fs = FileSystem.get(SparkSession.builder.getOrCreate.sparkContext.hadoopConfiguration)
    fs.listFiles(corpusDir, false)
      .toList
      .map(_.getPath)
      .filter(path => extensions.contains(FilenameUtils.getExtension(path.toString).toLowerCase))
  }

  /**
   * Trains an IDF [[PipelineModel]] based on the provided corpus files.
   * The model is saved to the specified saveDir. If saveDir is None the saving step is skipped.
   *
   * @param corpusFilePaths paths to files that form the training corpus
   * @param saveDir path to a directory to save the model in
   * @return a freshly trained IDF [[PipelineModel]]
   */
  private def train(corpusFilePaths: Seq[Path], saveDir: Option[Path]): PipelineModel = {
    val spark = SparkSession.builder.getOrCreate

    val trainingDF = spark.read.text(corpusFilePaths.map(_.toString): _*)
      .withColumnRenamed("value", "documentText")
      .withColumn("id", monotonically_increasing_id())
      .where(length(col("documentText")) > 0)

    // Annotations
    val documentAssemblerAnnotation = new DocumentAssembler()
      .setInputCol("documentText")
      .setOutputCol("document")
      .setCleanupMode("shrink")

    val documentNormalizerAnnotation = new DocumentNormalizer()
      .setInputCols("document")
      .setOutputCol("normalizedDocument")
      .setPatterns(Array("<[^>]*>"))
      .setPolicy("pretty_all")

    val sentenceAnnotation = new SentenceDetector()
      .setInputCols("normalizedDocument")
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
      .setOutputAsArray(true)
      .setCleanAnnotations(true) // Remove intermediate annotations

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
        documentNormalizerAnnotation,
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
    logger.info("Training pipeline model with the following stages: " + pipeline.getStages.mkString("\n"))
    val pipelineModel = pipeline.fit(trainingDF)

    // Save model
    if (saveDir.isDefined) {
      logger.info(s"Saving model to ${saveDir.get}")
      pipelineModel.write.overwrite.save(saveDir.get + "/" + LocalDateTime.now.toString.replace(":", "-"))
    } else {
      logger.info(s"Skipping model saving, no path defined.")
    }

    // Print vocabulary and IDF vector
    val idfModel: IDFModel = pipelineModel.stages.last.asInstanceOf[IDFModel]
    val vectorizerModel: CountVectorizerModel = pipelineModel.stages.apply(pipeline.getStages.length - 2).asInstanceOf[CountVectorizerModel]
    logger.info("Trained IDF model's vocabulary and scores: " + vectorizerModel.vocabulary.zip(idfModel.idf.toArray).mkString("\n"))

    pipelineModel
  }
}
