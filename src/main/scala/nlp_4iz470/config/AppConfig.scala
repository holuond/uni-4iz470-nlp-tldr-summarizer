package nlp_4iz470.config

import java.nio.file.Path

/**
 * A container for application configuration
 */
case class AppConfig(inputDocumentsDir: Path,
                     trainingCorpusDir: org.apache.hadoop.fs.Path,
                     modelsDir: org.apache.hadoop.fs.Path,
                     idfModelPath: Option[org.apache.hadoop.fs.Path]
                    )