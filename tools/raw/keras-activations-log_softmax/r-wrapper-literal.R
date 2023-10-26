#' Log-Softmax activation function.
#'
#' @description
#' Each input vector is handled independently.
#' The `axis` argument sets which axis of the input the function
#' is applied along.
#'
#' @param x Input tensor.
#' @param axis Integer, axis along which the softmax is applied.
#'
#' @export
#' @family activation functions
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/activations/log_softmax>
activation_log_softmax <-
structure(function (x, axis = -1L)
{
    args <- capture_args2(list(axis = as_axis))
    do.call(keras$activations$log_softmax, args)
}, py_function_name = "log_softmax")
