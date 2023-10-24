#' He normal initializer.
#'
#' @description
#' It draws samples from a truncated normal distribution centered on 0 with
#' `stddev = sqrt(2 / fan_in)` where `fan_in` is the number of input units in
#' the weight tensor.
#'
#' # Examples
#' ```python
#' # Standalone usage:
#' initializer = HeNormal()
#' values = initializer(shape=(2, 2))
#' ```
#'
#' ```python
#' # Usage in a Keras layer:
#' initializer = HeNormal()
#' layer = Dense(3, kernel_initializer=initializer)
#' ```
#'
#' # Reference
#' - [He et al., 2015](https://arxiv.org/abs/1502.01852)
#'
#' @param seed A Python integer or instance of
#' `keras.backend.SeedGenerator`.
#' Used to make the behavior of the initializer
#' deterministic. Note that an initializer seeded with an integer
#' or `None` (unseeded) will produce the same random values
#' across multiple calls. To get different random values
#' across multiple calls, use as seed an instance
#' of `keras.backend.SeedGenerator`.
#'
#' @export
#' @family initializer
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/initializers/HeNormal>
initializer_he_normal <-
function (seed = NULL)
{
    args <- capture_args2(list(seed = as_integer))
    do.call(keras$initializers$HeNormal, args)
}
