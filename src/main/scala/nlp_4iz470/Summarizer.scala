package nlp_4iz470

import com.johnsnowlabs.nlp.annotator.T5Transformer
import com.johnsnowlabs.nlp.{DocumentAssembler, Finisher}
import org.apache.log4j.Logger
import org.apache.spark.ml.feature.CountVectorizerModel
import org.apache.spark.ml.linalg.SparseVector
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.linalg.{Matrices, Matrix, SingularValueDecomposition, Vector, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.{array, collect_list, count, explode}

import scala.collection.mutable

object Summarizer {
  private val logger: Logger = Logger.getLogger(getClass.getName)

  /**
   * Takes an input document and returns a list of sentences of the original document as its summary.
   * The basic unit for the summary is a sentence (sentence boundaries are determined with a heuristic), and the amount
   * of returned sentences is reduced as per the selected reduction ratio.
   *
   * @param documentText     input document, e.g. a 5000 characters long news article
   * @param idfPipelineModel pretrained idf model
   * @param reductionRatio   determines how many sentences from the origina document will be kept in the summary
   * @return a summary of the original document
   */
  def summarizeDocument(documentText: String, idfPipelineModel: PipelineModel, reductionRatio: Double = 0.666667): String = {
    logger.info("************* Document Text *************")
    logger.info(documentText)

    val spark = SparkSession.builder.getOrCreate
    import spark.implicits._

    val row = List((1, documentText)) // the test DF has to contain one document contained in exactly one row for the annotations and logic to work properly
    val testDF = spark.createDataFrame(row).toDF("id", "documentText")

    // Verbose annotations
    idfPipelineModel.stages.apply(7).asInstanceOf[Finisher].setCleanAnnotations(false)

    // Score test data with IDF model
    val pipelineDF = idfPipelineModel.transform(testDF).cache

    // Extract vocabulary (mapping of: lemma id, lemma text)
    val vocabulary = idfPipelineModel.stages.apply(8).asInstanceOf[CountVectorizerModel].vocabulary.zipWithIndex
    val vocabularyDF = spark.createDataFrame(vocabulary).toDF("lemmaText", "lemmaVocabId")


    // Extract tf-idf scores (mapping of: lemma vocab id, lemma id (in test data), tfidf score)
    val tfIdfScoresVector = pipelineDF.select("idf").collect.head.get(0).asInstanceOf[SparseVector]
    val tfIdfScores: Array[(Int, Double, Int)] = tfIdfScoresVector.indices.zip(tfIdfScoresVector.values).zipWithIndex.map(x => (x._1._1, x._1._2, x._2))
    val tfIdfDf = spark.createDataFrame(tfIdfScores).toDF("lemmaVocabId_", "tfIdf", "lemmaId")

    // Extract key data from verbose pipeline output DF and transform so that there is exactly 1 row per lemma (each lemma occurrence)
    // Could be much more elegant with Spark 3.0 sql functions
    var lemmaTfIdfDF = pipelineDF
      .withColumn("lemma", explode($"lemma"))
      .select($"lemma.result" as "lemma",
        $"lemma.metadata.sentence" as "sentenceId",
        $"sentence" as "sentences")
      .withColumn("explodedSentence", explode($"sentences")) // explode sentences array
      .filter($"explodedSentence.metadata.sentence" === $"sentenceId") // .. and keep only 1 row per lemma with the correct sentence
      .withColumn("sentenceText", $"explodedSentence.result")
      .drop($"explodedSentence") // drop the rest of the exploded rows
      .drop($"sentences") // ... and the auxiliary sentences array
      .select($"lemma",
        $"sentenceId",
        $"sentenceText")
      .join(vocabularyDF, $"lemma" === $"lemmaText")
      .drop("lemmaText")
      .join(tfIdfDf, $"lemmaVocabId" === $"lemmaVocabId_")
      .drop($"lemmaVocabId_")
      .select($"lemmaId",
        $"lemma",
        $"lemmaVocabId",
        $"tfIdf",
        $"sentenceId" cast "int",
        $"sentenceText")
      .distinct // This squashes multiple occurrences of the same lemma in a given sentence, i.e. the selected strategy for the same lemma repeating in a sentence is keeping tf-idf as if it occurred just once

    // Some of the original sentences may not contain any meaningful lemmas, so the sentenceIds above might be e.g. 0, 1, 4, 5,
    // which would mess up the SVD indices, so we need an auxiliary monotonically increasing and consecutive sentence id system
    val auxSentenceIds: Seq[(Int, Int)] = lemmaTfIdfDF.select($"sentenceId").distinct.collect.toList.map(_.getInt(0)).sorted.zipWithIndex
    val auxSentenceIdsDf = spark.createDataFrame(auxSentenceIds).toDF("originalSentenceId", "auxSentenceId")
    lemmaTfIdfDF = lemmaTfIdfDF
      .join(auxSentenceIdsDf, $"sentenceId" === $"originalSentenceId")
      .drop("originalSentenceId", "sentenceId")
      .withColumnRenamed("auxSentenceId", "sentenceId")

    // Normalize tfidf by tf, effectively reducing to idf
    val lemmaCounts = lemmaTfIdfDF
      .groupBy("lemmaId")
      .agg(count("lemmaId") as "lemmaCount")
      .withColumnRenamed("lemmaId", "lemmaId_")
    lemmaTfIdfDF = lemmaTfIdfDF
      .join(lemmaCounts, $"lemmaId" === $"lemmaId_")
      .drop("lemmaId_")
      .withColumn("idf", $"tfIdf" / $"lemmaCount")

    logger.info("************* Annotated DataFrame (1 row per lemma occurrence) *************")
    lemmaTfIdfDF.cache
    lemmaTfIdfDF.printSchema
    lemmaTfIdfDF.sort($"tfIdf".desc).show(1000)

    // Create matrix [lemmaVocabId x sentenceId] with values being tf-idf scores
    // Prepare lemma/sentence arrays
    val lemmas = lemmaTfIdfDF.repartition(1).select("lemmaId", "lemma", "tfIdf", "idf").distinct
    logger.info("************* Lemmas: *************")
    lemmas.orderBy($"tfIdf".desc).show(1000, false)
    val sentences = lemmaTfIdfDF.repartition(1).select("sentenceId", "sentenceText").distinct.orderBy($"sentenceId")
    logger.info("************* Sentences: *************")
    sentences.show(1000, false)
    val sentenceIds: Array[Int] = sentences.select("sentenceId").distinct.collect.map(_.getInt(0)).sorted

    // Prepare sparse lemma vectors (lemma id -> sequence of tuples with sentence id and tf-idf score)
    // TODO tfidf is used as values of the SVD matrix, but idf could also be used or some function of tfidf/idf, like a logarithm etc.
    val vectors: Seq[Vector] = lemmaTfIdfDF.groupBy("lemmaId").agg(collect_list(array($"sentenceId" cast "double", $"tfIdf")) as "vectorContents")
      .collect
      .map(row => (row.getInt(0), row.getList[Seq[Double]](1).toArray.toSeq.map { case x: mutable.WrappedArray[java.lang.Double] => x.array.toSeq }.map { case Seq(a, b) => (a.toInt, b) }))
      .toList
      .sortBy(_._1)
      .map(x => Vectors.sparse(sentenceIds.length, x._2.toArray.asInstanceOf[Array[(Int, Double)]]))
    logger.info("************* Lemma Vectors (in prep for the matrix to decompose) *************")
    logger.info(vectors.mkString("\n"))

    // Pick num of concepts (this is a simplification, this should ideally be assessed and optimized for each usage)
    val numOfConcepts = sentenceIds.length
    logger.info("Num of concepts: " + numOfConcepts)

    // SVD
    logger.info("************* SVD *************")
    val rows: RDD[Vector] = spark.sparkContext.parallelize(vectors)
    val matrix: RowMatrix = new RowMatrix(rows)

    val svd: SingularValueDecomposition[RowMatrix, Matrix] = matrix.computeSVD(numOfConcepts, computeU = true)
    val u: RowMatrix = svd.U
    val s: Vector = svd.s
    val vt: Matrix = svd.V.transpose

    // Printing decomposition matrices
    logger.info("************* U: Lemma x Concept *************")
    logger.info("(How much does a given lemma contribute to a given concept)")
    logger.info("U: Num rows = " + u.numRows)
    logger.info("U: Num cols = " + u.numCols)
    // Spark SVD implementation requests RowMatrix for matrix U, but it doesn't keep meaningful row indices
    // logger.info(lemmas.repartition(1).orderBy("lemmaId").collect.map(_.getString(1)).zip(u.rows.collect).foreach(println))

    logger.info("************* S: Concept priority *************")
    logger.info("S: Size = " + s.size)
    logger.info(s)

    logger.info("************* VT: Concept x Sentence *************")
    logger.info("VT: Num rows = " + vt.numRows)
    logger.info("VT: Num cols = " + vt.numCols)
    logger.info(vt.toString(1000, 1000))

    // Score sentences
    val rawSentenceConceptImportance: Seq[(Double, Int, String)] =
      Matrices.diag(s) // account for concept priority (multiply values of each concept by it's priority)
        .multiply(vt.asInstanceOf[org.apache.spark.mllib.linalg.DenseMatrix])
        .transpose
        .rowIter
        .toList
        .map(sentenceVec => math.sqrt(Vectors.sqdist(sentenceVec, Vectors.dense(1.to(numOfConcepts).toArray.map(x => 0.0))))) // calculate euclidean vector length
        .zip(sentences.collect.sortBy(_.getInt(0)))
        .sortBy(_._1)
        .map { case (score, sentenceIdAndText) => (score, sentenceIdAndText.getInt(0), sentenceIdAndText.getString(1)) }

    // Normalize:
    // penalize with sentence length (divide score by lemma count in a given sentence)
    // & normalize to interval <0,1>
    val lemmaCountsInSentence = lemmaTfIdfDF
      .groupBy("sentenceId")
      .agg(count("sentenceId") as "lemmaCountInSentence")
      .orderBy("sentenceId")
      .collect
      .map(_.getLong(1).toInt)
    val penalizedSentenceConceptImportance: Seq[(Double, Int, String)] =
      rawSentenceConceptImportance
        .zip(lemmaCountsInSentence)
        .map { case ((score, id, sentence), lemmaCountInSentence) => (score / lemmaCountInSentence, id, sentence) }
        .sortBy(_._1)
        .reverse
    val maxScore: Double = penalizedSentenceConceptImportance.map(_._1).max
    val normalizedSentenceConceptImportance: Seq[(Double, Int, String)] =
      penalizedSentenceConceptImportance
        .map { case (score, id, sentence) => (score / maxScore, id, sentence) }

    // Print RESULTS
    // TODO use LSA together with tf-idf for keyword priority ordering
    val numOfSentencesToKeep: Int = math.round(sentenceIds.length * (1.0 - reductionRatio)).toInt
    logger.info("\n************* Keywords (only tf-idf based) *************\n")
    logger.info(lemmaTfIdfDF.select($"lemma", $"tfIdf").distinct.repartition(1).sort($"tfIdf".desc)
      .collect.map(_.getString(0)).take(numOfSentencesToKeep).mkString(", "))

    logger.info("\n************* Scored Sentences (based on tf-idf and LSA, ordered by priority) *************\n")
    logger.info("(score, sentence id, sentence text)")
    logger.info(normalizedSentenceConceptImportance.mkString("\n"))

    val summary: String = normalizedSentenceConceptImportance.sortBy(_._1).reverse.take(numOfSentencesToKeep).sortBy(_._2).map(_._3).mkString("\n")
    logger.info(s"\n************* Document Summary (reduced from ${sentenceIds.length} sentences to $numOfSentencesToKeep) *************\n")
    logger.info(summary)

    summary
  }

  def summarizeWithT5(documentText: String): Unit = {
    logger.info("************* Summarize document with JohnSnowLabs T5 Transformer *************")
    logger.info(documentText)

    val spark = SparkSession.builder.getOrCreate

    val documentAssemblerAnnotation = new DocumentAssembler()
      .setInputCol("documentText")
      .setOutputCol("document")
      .setCleanupMode("shrink")

    val summarizeAnnotation = T5Transformer
      .pretrained("t5_small", "en")
      .setTask("summarize:")
      .setMaxOutputLength(600)
      .setInputCols(Array("document"))
      .setOutputCol("summary")

    val pipeline = new Pipeline()
      .setStages(Array(
        documentAssemblerAnnotation,
        summarizeAnnotation
      ))

    val row = List((1, documentText))
    val df = spark.createDataFrame(row).toDF("id", "documentText")

    val model = pipeline.fit(df)
    val result = model.transform(df)

    result.show(1000, truncate = false)
  }
}