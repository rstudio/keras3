#' Return evenly spaced values within a given interval.
#'
#' @description
#' `arange` can be called with a varying number of positional arguments:
#' * `arange(stop)`: Values are generated within the half-open interval
#'     `[0, stop)` (in other words, the interval including start but excluding
#'     stop).
#' * `arange(start, stop)`: Values are generated within the half-open interval
#'     `[start, stop)`.
#' * `arange(start, stop, step)`: Values are generated within the half-open
#'     interval `[start, stop)`, with spacing between values given by step.
#'
#' # Returns
#' Tensor of evenly spaced values.
#' For floating point arguments, the length of the result is
#' `ceil((stop - start)/step)`. Because of floating point overflow, this
#' rule may result in the last element of out being greater than stop.
#'
#' # Examples
#' ```python
#' keras.ops.arange(3)
#' # array([0, 1, 2], dtype=int32)
#' ```
#'
#' ```python
#' keras.ops.arange(3.0)
#' # array([0., 1., 2.], dtype=float32)
#' ```
#'
#' ```python
#' keras.ops.arange(3, 7)
#' # array([3, 4, 5, 6], dtype=int32)
#' ```
#'
#' ```python
#' keras.ops.arange(3, 7, 2)
#' # array([3, 5], dtype=int32)
#' ```
#'
#' @param start Integer or real, representing the start of the interval. The
#'     interval includes this value.
#' @param stop Integer or real, representing the end of the interval. The
#'     interval does not include this value, except in some cases where
#'     `step` is not an integer and floating point round-off affects the
#'     lenght of `out`. Defaults to `None`.
#' @param step Integer or real, represent the spacing between values. For any
#'     output `out`, this is the distance between two adjacent values,
#'     `out[i+1] - out[i]`. The default step size is 1. If `step` is
#'     specified as a position argument, `start` must also be given.
#' @param dtype The type of the output array. If `dtype` is not given, infer the
#'     data type from the other input arguments.
#'
#' @export
#' @family ops
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/arange>
k_arange <-
function (start, stop = NULL, step = 1L, dtype = NULL)
{
    args <- capture_args2(list(start = as_integer, stop = as_integer,
        step = as_integer))
    do.call(keras$ops$arange, args)
}
