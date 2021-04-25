package nlp_4iz470.config

import org.apache.log4j.Logger
import pureconfig.ConfigSource
// necessary for pureconfig: import pureconfig.generic.auto._
import pureconfig.generic.auto._

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
      logger.info(s"Auto-reading config file with PureConfig by scanning default configuration file locations.")
      val appConfig: AppConfig = ConfigSource.default.at("app").loadOrThrow[AppConfig]

      logger.info("Successfully loaded an AppConfig instance")
      appConfig
    }
  }
}
