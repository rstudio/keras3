#' Random uniform initializer.
#'
#' @description
#' Draws samples from a uniform distribution for given parameters.
#'
#' # Examples
#' ```python
#' # Standalone usage:
#' initializer = RandomUniform(minval=0.0, maxval=1.0)
#' values = initializer(shape=(2, 2))
#' ```
#'
#' ```python
#' # Usage in a Keras layer:
#' initializer = RandomUniform(minval=0.0, maxval=1.0)
#' layer = Dense(3, kernel_initializer=initializer)
#' ```
#'
#' @param minval A python scalar or a scalar keras tensor. Lower bound of the
#'     range of random values to generate (inclusive).
#' @param maxval A python scalar or a scalar keras tensor. Upper bound of the
#'     range of random values to generate (exclusive).
#' @param seed A Python integer or instance of
#'     `keras.backend.SeedGenerator`.
#'     Used to make the behavior of the initializer
#'     deterministic. Note that an initializer seeded with an integer
#'     or `None` (unseeded) will produce the same random values
#'     across multiple calls. To get different random values
#'     across multiple calls, use as seed an instance
#'     of `keras.backend.SeedGenerator`.
#'
#' @export
#' @family initializer
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/initializers/RandomUniform>
initializer_random_uniform <-
function (minval = -0.05, maxval = 0.05, seed = NULL)
{
    args <- capture_args2(list(seed = as_integer))
    do.call(keras$initializers$RandomUniform, args)
}
