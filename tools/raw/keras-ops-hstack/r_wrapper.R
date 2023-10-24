#' Stack tensors in sequence horizontally (column wise).
#'
#' @description
#' This is equivalent to concatenation along the first axis for 1-D tensors,
#' and along the second axis for all other tensors.
#'
#' # Returns
#'     The tensor formed by stacking the given tensors.
#'
#' @param xs Sequence of tensors.
#'
#' @export
#' @family ops
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/hstack>
k_hstack <-
function (xs)
keras$ops$hstack(xs)
