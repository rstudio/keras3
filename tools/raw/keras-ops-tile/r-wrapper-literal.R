#' Repeat `x` the number of times given by `repeats`.
#'
#' @description
#' If `repeats` has length `d`, the result will have dimension of
#' `max(d, x.ndim)`.
#'
#' If `x.ndim < d`, `x` is promoted to be d-dimensional by prepending
#' new axes.
#'
#' If `x.ndim > d`, `repeats` is promoted to `x.ndim` by prepending 1's to it.
#'
#' # Returns
#'     The tiled output tensor.
#'
#' @param x Input tensor.
#' @param repeats The number of repetitions of `x` along each axis.
#'
#' @export
#' @family ops
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/tile>
k_tile <-
function (x, repeats)
keras$ops$tile(x, repeats)
