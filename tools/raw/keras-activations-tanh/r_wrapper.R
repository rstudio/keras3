#' Hyperbolic tangent activation function.
#'
#' @description
#' It is defined as:
#' `tanh(x) = sinh(x) / cosh(x)`, i.e.
#' `tanh(x) = ((exp(x) - exp(-x)) / (exp(x) + exp(-x)))`.
#'
#' @param x Input tensor.
#'
#' @export
#' @family activation functions
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/activations/tanh>
activation_tanh <-
structure(function (x)
{
    args <- capture_args2(NULL)
    do.call(keras$activations$tanh, args)
}, py_function_name = "tanh")
