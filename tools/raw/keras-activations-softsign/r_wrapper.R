#' Softsign activation function.
#'
#' @description
#' Softsign is defined as: `softsign(x) = x / (abs(x) + 1)`.
#'
#' @param x Input tensor.
#'
#' @export
#' @family activation functions
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/activations/softsign>
activation_softsign <-
structure(function (x)
{
    args <- capture_args2(NULL)
    do.call(keras$activations$softsign, args)
}, py_function_name = "softsign")
