#' Return `x[key]`.
#'
#' @description
#'
#' @param x see description
#' @param key see description
#'
#' @export
#' @family ops
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/get_item>
k_get_item <-
function (x, key)
keras$ops$get_item(x, key)
