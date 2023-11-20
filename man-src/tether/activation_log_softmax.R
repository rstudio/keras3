#' Log-Softmax activation function.
#'
#' @description
#' Each input vector is handled independently.
#' The `axis` argument sets which axis of the input the function
#' is applied along.
#'
#' @param x
#' Input tensor.
#'
#' @param axis
#' Integer, axis along which the softmax is applied.
#'
#' @export
#' @family activations
#' @seealso
#' + <https:/keras.io/keras_core/api/layers/activations#logsoftmax-function>
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/activations/log_softmax>
activation_log_softmax <-
function (x, axis = -1L)
{
}
