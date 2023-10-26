#' Initializer that generates tensors with constant values.
#'
#' @description
#' Only scalar values are allowed.
#' The constant value provided must be convertible to the dtype requested
#' when calling the initializer.
#'
#' # Examples
#' ```python
#' # Standalone usage:
#' initializer = Constant(10.)
#' values = initializer(shape=(2, 2))
#' ```
#'
#' ```python
#' # Usage in a Keras layer:
#' initializer = Constant(10.)
#' layer = Dense(3, kernel_initializer=initializer)
#' ```
#'
#' @param value A Python scalar.
#'
#' @export
#' @family initializer
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/initializers/constant>
initializer_constant <-
function (value = 0)
{
    args <- capture_args2(NULL)
    do.call(keras$initializers$constant, args)
}
