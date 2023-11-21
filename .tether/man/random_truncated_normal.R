#' Draw samples from a truncated normal distribution.
#'
#' @description
#' The values are drawn from a normal distribution with specified mean and
#' standard deviation, discarding and re-drawing any samples that are more
#' than two standard deviations from the mean.
#'
#' @param shape
#' The shape of the random values to generate.
#'
#' @param mean
#' Float, defaults to 0. Mean of the random values to generate.
#'
#' @param stddev
#' Float, defaults to 1. Standard deviation of the random values
#' to generate.
#'
#' @param dtype
#' Optional dtype of the tensor. Only floating point types are
#' supported. If not specified, `keras.config.floatx()` is used,
#' which defaults to `float32` unless you configured it otherwise (via
#' `keras.config.set_floatx(float_dtype)`)
#'
#' @param seed
#' A Python integer or instance of
#' `keras.random.SeedGenerator`.
#' Used to make the behavior of the initializer
#' deterministic. Note that an initializer seeded with an integer
#' or None (unseeded) will produce the same random values
#' across multiple calls. To get different random values
#' across multiple calls, use as seed an instance
#' of `keras.random.SeedGenerator`.
#'
#' @export
#' @family random
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/random/truncated_normal>
random_truncated_normal <-
function (shape, mean = 0, stddev = 1, dtype = NULL, seed = NULL)
{
}
