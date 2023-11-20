#' Draw random samples from the Gamma distribution.
#'
#' @param shape
#' The shape of the random values to generate.
#'
#' @param alpha
#' Float, the parameter of the distribution.
#'
#' @param dtype
#' Optional dtype of the tensor. Only floating point types are
#' supported. If not specified, `keras.config.floatx()` is used,
#' which defaults to `float32` unless you configured it otherwise (via
#' `keras.config.set_floatx(float_dtype)`).
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
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/random/gamma>
random_gamma <-
function (shape, alpha, dtype = NULL, seed = NULL)
{
}
