#' Linear activation function (pass-through).
#'
#' @description
#' A "linear" activation is an identity function:
#' it returns the input, unmodified.
#'
#' @param x Input tensor.
#'
#' @export
#' @family activation functions
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/activations/linear>
activation_linear <-
structure(function (x)
{
    args <- capture_args2(NULL)
    do.call(keras$activations$linear, args)
}, py_function_name = "linear")
