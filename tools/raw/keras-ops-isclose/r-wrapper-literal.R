#' Return whether two tensors are element-wise almost equal.
#'
#' @description
#'
#' # Returns
#'     Output boolean tensor.
#'
#' @param x1 First input tensor.
#' @param x2 Second input tensor.
#'
#' @export
#' @family ops
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/isclose>
k_isclose <-
function (x1, x2)
keras$ops$isclose(x1, x2)
