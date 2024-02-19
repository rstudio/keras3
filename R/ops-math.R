#' Computes the inverse error function of `x`, element-wise.
#'
#' @description
#'
#' # Examples
#' ```{r}
#' x <- op_array(c(-0.5, -0.2, -0.1, 0.0, 0.3))
#' op_erfinv(x)
#' ```
#'
#' @returns
#' A tensor with the same dtype as `x`.
#'
#' @param x
#' Input tensor.
#'
#' @export
#' @family math ops
#' @family ops
#' @tether keras.ops.erfinv
# @seealso
# + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/erfinv>
op_erfinv <-
function (x)
keras$ops$erfinv(x)
