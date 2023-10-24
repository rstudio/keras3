#' Gives a new shape to a tensor without changing its data.
#'
#' @description
#'
#' # Returns
#'     The reshaped tensor.
#'
#' @param x Input tensor.
#' @param new_shape The new shape should be compatible with the original shape.
#'     One shape dimension can be -1 in which case the value is
#'     inferred from the length of the array and remaining dimensions.
#'
#' @export
#' @family ops
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/reshape>
k_reshape <-
function (x, new_shape)
keras$ops$reshape(x, new_shape)
