#' Normalizes `x` by `mean` and `variance`.
#'
#' @description
#' This op is typically used by the batch normalization step in a neural
#' network. It normalizes the input tensor along the given axis.
#'
#' # Examples
#' ```python
#' x = keras.ops.convert_to_tensor(
#'     [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
#' )
#' keras.ops.batch_normalization(
#'     x,
#'     mean=[0.4, 0.5, 0.6],
#'     variance=[0.67, 0.67, 0.67],
#'     axis=-1
#' )
#' # array([[-3.6624e-01, -3.6624e-01, -3.6624e-01],
#' #        [-4.6445e-09,  0.0000e+00, -1.8578e-08],
#' #        [ 3.6624e-01,  3.6624e-01,  3.6624e-01]])
#' ```
#'
#' @returns
#' The normalized tensor.
#'
#' @param x
#' Input tensor.
#'
#' @param mean
#' A mean vector of the same length as the `axis` dimension of the
#' input thensor.
#'
#' @param variance
#' A variance vector of the same length as the `axis` dimension
#' of the input tensor.
#'
#' @param axis
#' Integer, the axis that should be normalized.
#'
#' @param offset
#' An offset vector of the same length as the `axis` dimension of
#' the input tensor. If not `None`, `offset` is added to the normalized
#' tensor. Defaults to `None`.
#'
#' @param scale
#' A scale vector of the same length as the `axis` dimension of the
#' input tensor. If not `None`, the normalized tensor is multiplied by
#' `scale`. Defaults to `None`.
#'
#' @param epsilon
#' Small float added to variance to avoid dividing by zero.
#' Defaults to 1e-3.
#'
#' @export
#' @tether keras.ops.batch_normalization
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/batch_normalization>
op_batch_normalization <-
function (x, mean, variance, axis, offset = NULL, scale = NULL,
    epsilon = 0.001)
{
    args <- capture_args(list(axis = as_axis))
    do.call(keras$ops$batch_normalization, args)
}

#' Normalizes `x` over the specified axis.
#'
#' @description
#' It is defined as: `normalize(x) = x / max(norm(x), epsilon)`.
#'
#' # Examples
#' ```python
#' x = keras.ops.convert_to_tensor([[1, 2, 3], [4, 5, 6]])
#' x_norm = keras.ops.math.normalize(x)
#' print(x_norm)
#' # array([[0.26726124 0.5345225  0.8017837 ]
#' #        [0.45584232 0.5698029  0.68376344]], shape=(2, 3), dtype=float32)
#' ```
#'
#' @returns
#' The normalized array.
#'
#' @param x
#' Input tensor.
#'
#' @param axis
#' The axis or axes along which to perform normalization.
#' Default to -1.
#'
#' @param order
#' The exponent value in the norm formulation.
#' Defaults to 2.
#'
#' @export
#' @tether keras.ops.normalize
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/normalize>
op_normalize <-
function (x, axis = -1L, order = 2L)
{
    args <- capture_args(list(axis = as_axis, order = as_integer))
    do.call(keras$ops$normalize, args)
}