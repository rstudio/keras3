#' Initializer that generates the identity matrix.
#'
#' @description
#' Only usable for generating 2D matrices.
#'
#' # Examples
#' ```python
#' # Standalone usage:
#' initializer = Identity()
#' values = initializer(shape=(2, 2))
#' ```
#'
#' ```python
#' # Usage in a Keras layer:
#' initializer = Identity()
#' layer = Dense(3, kernel_initializer=initializer)
#' ```
#'
#' @param gain Multiplicative factor to apply to the identity matrix.
#'
#' @export
#' @family initializer
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/initializers/identity>
initializer_identity <-
function (gain = 1)
{
    args <- capture_args2(NULL)
    do.call(keras$initializers$identity, args)
}
