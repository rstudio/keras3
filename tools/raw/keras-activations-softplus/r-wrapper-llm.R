#' Softplus activation function.
#'
#' @description
#' It is defined as: `softplus(x) = log(exp(x) + 1)`.
#'
#' @param x Input tensor.
#'
#' @export
#' @family activation functions
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/activations/softplus>
activation_softplus <-
structure(function (x)
{
    args <- capture_args2(NULL)
    do.call(keras$activations$softplus, args)
}, py_function_name = "softplus")
