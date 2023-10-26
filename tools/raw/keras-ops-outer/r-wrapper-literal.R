#' Compute the outer product of two vectors.
#'
#' @description
#' Given two vectors `x1` and `x2`, the outer product is:
#'
#' ```
#' out[i, j] = x1[i] * x2[j]
#' ```
#'
#' # Returns
#'     Outer product of `x1` and `x2`.
#'
#' @param x1 First input tensor.
#' @param x2 Second input tensor.
#'
#' @export
#' @family ops
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/outer>
k_outer <-
function (x1, x2)
keras$ops$outer(x1, x2)
