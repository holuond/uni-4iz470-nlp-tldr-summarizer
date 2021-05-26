package nlp_4iz470.config

/**
 * A container for application configuration
 */
case class AppConfig(trainingCorpusDir: org.apache.hadoop.fs.Path,
                     modelsDir: org.apache.hadoop.fs.Path,
                     idfModelPath: Option[org.apache.hadoop.fs.Path]
                    )