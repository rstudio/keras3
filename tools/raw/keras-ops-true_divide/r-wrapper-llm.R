#' Alias for `keras.ops.divide`.
#'
#' @description
#'
#' @param x1 see description
#' @param x2 see description
#'
#' @export
#' @family ops
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/true_divide>
k_true_divide <-
function (x1, x2)
keras$ops$true_divide(x1, x2)
