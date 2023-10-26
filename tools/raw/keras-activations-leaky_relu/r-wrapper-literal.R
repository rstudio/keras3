#' Leaky relu activation function.
#'
#' @description
#'
#' @param x Input tensor.
#' @param negative_slope A `float` that controls the slope
#'     for values lower than the threshold.
#'
#' @export
#' @family activation functions
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/activations/leaky_relu>
activation_leaky_relu <-
structure(function (x, negative_slope = 0.2)
{
    args <- capture_args2(NULL)
    do.call(keras$activations$leaky_relu, args)
}, py_function_name = "leaky_relu")
