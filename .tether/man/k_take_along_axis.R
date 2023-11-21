#' Select values from `x` at the 1-D `indices` along the given axis.
#'
#' @returns
#'     The corresponding tensor of values.
#'
#' @param x
#' Source tensor.
#'
#' @param indices
#' The indices of the values to extract.
#'
#' @param axis
#' The axis over which to select values. By default, the flattened
#' input tensor is used.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @seealso
#' + <https:/keras.io/keras_core/api/ops/numpy#takealongaxis-function>
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/take_along_axis>
k_take_along_axis <-
function (x, indices, axis = NULL)
{
}
