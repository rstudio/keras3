#' Exponential activation function.
#'
#' @description
#'
#' @param x Input tensor.
#'
#' @export
#' @family activation functions
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/activations/exponential>
activation_exponential <-
structure(function (x)
{
    args <- capture_args2(NULL)
    do.call(keras$activations$exponential, args)
}, py_function_name = "exponential")
