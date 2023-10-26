#' Initializer that generates tensors initialized to 1.
#'
#' @description
#' Also available via the shortcut function `ones`.
#'
#' # Examples
#' ```python
#' # Standalone usage:
#' initializer = Ones()
#' values = initializer(shape=(2, 2))
#' ```
#'
#' ```python
#' # Usage in a Keras layer:
#' initializer = Ones()
#' layer = Dense(3, kernel_initializer=initializer)
#' ```
#'
#' @export
#' @family initializer
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/initializers/ones>
initializer_ones <-
function ()
{
    args <- capture_args2(NULL)
    do.call(keras$initializers$ones, args)
}
