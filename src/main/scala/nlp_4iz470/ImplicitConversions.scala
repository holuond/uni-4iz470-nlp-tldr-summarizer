package nlp_4iz470

import org.apache.hadoop.fs.{Path, RemoteIterator}

import scala.language.implicitConversions

object ImplicitConversions {

  /**
   * Converts RemoteIterator from Hadoop to Scala Iterator that provides all the familiar functions such as map,
   * filter, foreach, etc.
   *
   * @param underlying The RemoteIterator that needs to be wrapped
   * @tparam T Items inside the iterator
   * @see [[https://gist.github.com/vsimko/04fd6d289bac4b7c66a7e870c2ffc718 source]]
   * @return Standard Scala Iterator
   */
  implicit def convertToScalaIterator[T](underlying: RemoteIterator[T]): Iterator[T] = {
    case class Wrapper(underlying: RemoteIterator[T]) extends Iterator[T] {
      override def hasNext: Boolean = underlying.hasNext

      override def next: T = underlying.next
    }
    Wrapper(underlying)
  }

  /**
   * Converts a Hadoop path to a Scala String.
   *
   * @return path String
   */
  implicit def hadoopPath2String(a: Path): String = {
    a.toString
  }
}
