#' Append tensor `x2` to the end of tensor `x1`.
#'
#' @description
#'
#' # Examples
#' ```python
#' x1 = keras.ops.convert_to_tensor([1, 2, 3])
#' x2 = keras.ops.convert_to_tensor([[4, 5, 6], [7, 8, 9]])
#' keras.ops.append(x1, x2)
#' # array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int32)
#' ```
#'
#' When `axis` is specified, `x1` and `x2` must have compatible shapes.
#' ```python
#' x1 = keras.ops.convert_to_tensor([[1, 2, 3], [4, 5, 6]])
#' x2 = keras.ops.convert_to_tensor([[7, 8, 9]])
#' keras.ops.append(x1, x2, axis=0)
#' # array([[1, 2, 3],
#' #         [4, 5, 6],
#' #         [7, 8, 9]], dtype=int32)
#' x3 = keras.ops.convert_to_tensor([7, 8, 9])
#' keras.ops.append(x1, x3, axis=0)
#' # Traceback (most recent call last):
#' #     ...
#' # TypeError: Cannot concatenate arrays with different numbers of
#' # dimensions: got (2, 3), (3,).
#' ```
#'
#' @returns
#' A tensor with the values of `x2` appended to `x1`.
#'
#' @param x1
#' First input tensor.
#'
#' @param x2
#' Second input tensor.
#'
#' @param axis
#' Axis along which tensor `x2` is appended to tensor `x1`.
#' If `None`, both tensors are flattened before use.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @seealso
#' + <https:/keras.io/keras_core/api/ops/numpy#append-function>
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/append>
k_append <-
function (x1, x2, axis = NULL)
{
}
