#' Returns the cross product of two (arrays of) vectors.
#'
#' @description
#' The cross product of `x1` and `x2` in R^3 is a vector
#' perpendicular to both `x1` and `x2`. If `x1` and `x2` are arrays of
#' vectors, the vectors are defined by the last axis of `x1` and `x2`
#' by default, and these axes can have dimensions 2 or 3.
#'
#' Where the dimension of either `x1` or `x2` is 2, the third component of
#' the input vector is assumed to be zero and the cross product calculated
#' accordingly.
#'
#' In cases where both input vectors have dimension 2, the z-component of
#' the cross product is returned.
#'
#' # Note
#' Torch backend does not support two dimensional vectors, or the
#' arguments `axisa`, `axisb` and `axisc`. Use `axis` instead.
#'
#' # Returns
#'     Vector cross product(s).
#'
#' @param x1 Components of the first vector(s).
#' @param x2 Components of the second vector(s).
#' @param axisa Axis of `x1` that defines the vector(s). Defaults to `-1`.
#' @param axisb Axis of `x2` that defines the vector(s). Defaults to `-1`.
#' @param axisc Axis of the result containing the cross product vector(s).
#'     Ignored if both input vectors have dimension 2, as the return is
#'     scalar. By default, the last axis.
#' @param axis If defined, the axis of `x1`, `x2` and the result that
#'     defines the vector(s) and cross product(s). Overrides `axisa`,
#'     `axisb` and `axisc`.
#'
#' @export
#' @family ops
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/cross>
k_cross <-
function (x1, x2, axisa = -1L, axisb = -1L, axisc = -1L, axis = NULL)
{
    args <- capture_args2(list(axisa = as_integer, axisb = as_integer,
        axisc = as_integer, axis = as_axis))
    do.call(keras$ops$cross, args)
}
