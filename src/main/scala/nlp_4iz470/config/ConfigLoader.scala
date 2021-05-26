package nlp_4iz470.config

import org.apache.hadoop.fs
import org.apache.log4j.Logger
import pureconfig.{ConfigReader, ConfigSource}

import java.io.File
import java.nio.file.Paths
import scala.util.Try

/**
 * Reads input parameters from the command line,
 * extracts the HOCON application config into [[AppConfig]] case class
 */
object ConfigLoader {
  private val logger: Logger = Logger.getLogger(getClass.getName)

  /**
   * Takes in command line args and parses it into the [[AppConfig]] case class container
   *
   * @see [[https://pureconfig.github.io/docs/ PureConfig documentation]]
   * @return Try wrapped [[AppConfig]] container
   */
  def loadAppConfig(): Try[AppConfig] = {
    Try {
      import pureconfig.generic.auto._
      import CustomConfigImplicits._

      logger.info(s"Auto-reading config file with PureConfig by scanning default configuration file locations.")
      val appConfig: AppConfig = ConfigSource.default.at("app").loadOrThrow[AppConfig]

      logger.info("Successfully loaded an AppConfig instance")
      appConfig
    }
  }

  /**
   * Allows PureConfig to implicitly convert primitive types found in the HOCON config file
   * to non-primitive types that the [[AppConfig]] case class expects.
   */
  object CustomConfigImplicits {
    // HOCON String --> java.nio.Path
    implicit val pathReader: ConfigReader[java.nio.file.Path] = ConfigReader[String].map(Paths.get(_))

    // HOCON String --> java.io.File
    implicit val fileReader: ConfigReader[File] = ConfigReader[String].map(new File(_))

    // HOCON String --> org.apache.hadoop.fs.Path
    implicit val hadoopPathReader: ConfigReader[org.apache.hadoop.fs.Path] = ConfigReader[String].map(new fs.Path(_))
  }
}

