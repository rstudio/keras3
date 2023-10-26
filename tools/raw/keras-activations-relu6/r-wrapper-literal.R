#' Relu6 activation function.
#'
#' @description
#' It's the ReLU function, but truncated to a maximum value of 6.
#'
#' @param x Input tensor.
#'
#' @export
#' @family activation functions
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/activations/relu6>
activation_relu_6 <-
structure(function (x)
{
    args <- capture_args2(NULL)
    do.call(keras$activations$relu6, args)
}, py_function_name = "relu6")
