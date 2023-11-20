#' Test whether any array element along a given axis evaluates to `True`.
#'
#' @description
#'
#' # Examples
#' ```python
#' x = keras.ops.convert_to_tensor([True, False])
#' keras.ops.any(x)
#' # array(True, shape=(), dtype=bool)
#' ```
#'
#' ```python
#' x = keras.ops.convert_to_tensor([[True, False], [True, True]])
#' keras.ops.any(x, axis=0)
#' # array([ True  True], shape=(2,), dtype=bool)
#' ```
#'
#' `keepdims=True` outputs a tensor with dimensions reduced to one.
#' ```python
#' x = keras.ops.convert_to_tensor([[True, False], [True, True]])
#' keras.ops.all(x, keepdims=True)
#' # array([[False]], shape=(1, 1), dtype=bool)
#' ```
#'
#' @returns
#' The tensor containing the logical OR reduction over the `axis`.
#'
#' @param x
#' Input tensor.
#'
#' @param axis
#' An integer or tuple of integers that represent the axis along
#' which a logical OR reduction is performed. The default
#' (`axis=None`) is to perform a logical OR over all the dimensions
#' of the input array. `axis` may be negative, in which case it counts
#' for the last to the first axis.
#'
#' @param keepdims
#' If `True`, axes which are reduced are left in the result as
#' dimensions with size one. With this option, the result will
#' broadcast correctly against the input array. Defaults to`False`.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @seealso
#' + <https:/keras.io/keras_core/api/ops/numpy#any-function>
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/any>
k_any <-
function (x, axis = NULL, keepdims = FALSE)
{
}
