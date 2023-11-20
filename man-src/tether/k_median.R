#' Compute the median along the specified axis.
#'
#' @returns
#'     The output tensor.
#'
#' @param x
#' Input tensor.
#'
#' @param axis
#' Axis or axes along which the medians are computed. Defaults to
#' `axis=None` which is to compute the median(s) along a flattened
#' version of the array.
#'
#' @param keepdims
#' If this is set to `True`, the axes which are reduce
#' are left in the result as dimensions with size one.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/median>
k_median <-
function (x, axis = NULL, keepdims = FALSE)
{
}
