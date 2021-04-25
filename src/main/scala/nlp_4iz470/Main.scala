package nlp_4iz470

import nlp_4iz470.config.{AppConfig, ConfigLoader}
import org.apache.log4j.Logger

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

    // TODO Continue :)
  }
}
