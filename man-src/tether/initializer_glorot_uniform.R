#' The Glorot uniform initializer, also called Xavier uniform initializer.
#'
#' @description
#' Draws samples from a uniform distribution within `[-limit, limit]`, where
#' `limit = sqrt(6 / (fan_in + fan_out))` (`fan_in` is the number of input
#' units in the weight tensor and `fan_out` is the number of output units).
#'
#' # Examples
#' ```python
#' # Standalone usage:
#' initializer = GlorotUniform()
#' values = initializer(shape=(2, 2))
#' ```
#'
#' ```python
#' # Usage in a Keras layer:
#' initializer = GlorotUniform()
#' layer = Dense(3, kernel_initializer=initializer)
#' ```
#'
#' # Reference
#' - [Glorot et al., 2010](http://proceedings.mlr.press/v9/glorot10a.html)
#'
#' @param seed
#' A Python integer or instance of
#' `keras.backend.SeedGenerator`.
#' Used to make the behavior of the initializer
#' deterministic. Note that an initializer seeded with an integer
#' or `None` (unseeded) will produce the same random values
#' across multiple calls. To get different random values
#' across multiple calls, use as seed an instance
#' of `keras.backend.SeedGenerator`.
#'
#' @export
#' @family random initializers
#' @family initializers
#' @seealso
#' + <https:/keras.io/api/layers/initializers#glorotuniform-class>
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/initializers/GlorotUniform>
initializer_glorot_uniform <-
function (seed = NULL)
{
}
