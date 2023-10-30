#' Inverse sine, element-wise.
#'
#' @description
#' Returns a tensor of the inverse sine of each element in `x`, in radians and in
#' the closed interval `[-pi/2, pi/2]`.
#'
#' @param x Input tensor.
#' @export
#' @family ops
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/arcsin>
#'
#' @examples
#' x <- tf$convert_to_tensor(c(1, -1, 0))
#' tf$arcsin(x)
#' # array([ 1.5707964, -1.5707964,  0.], dtype=float32)
k_arcsin <-
function (x)
keras$ops$arcsin(x)
