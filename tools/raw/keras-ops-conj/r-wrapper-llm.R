#' Shorthand for `keras.ops.conjugate`.
#'
#' @description
#'
#' @param x see description
#'
#' @export
#' @family ops
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/conj>
k_conj <-
function (x)
keras$ops$conj(x)
