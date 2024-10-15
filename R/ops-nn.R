#' Normalizes `x` by `mean` and `variance`.
#'
#' @description
#' This op is typically used by the batch normalization step in a neural
#' network. It normalizes the input tensor along the given axis.
#'
#' # Examples
#' ```{r}
#' x <- op_convert_to_tensor(rbind(c(0.1, 0.2, 0.3),
#'                                 c(0.4, 0.5, 0.6),
#'                                 c(0.7, 0.8, 0.9)))
#' op_batch_normalization(
#'   x,
#'   mean = c(0.4, 0.5, 0.6),
#'   variance = c(0.67, 0.67, 0.67),
#'   axis = -1
#' )
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
#' the input tensor. If not `NULL`, `offset` is added to the normalized
#' tensor. Defaults to `NULL`.
#'
#' @param scale
#' A scale vector of the same length as the `axis` dimension of the
#' input tensor. If not `NULL`, the normalized tensor is multiplied by
#' `scale`. Defaults to `NULL`.
#'
#' @param epsilon
#' Small float added to variance to avoid dividing by zero.
#' Defaults to 1e-3.
#'
#' @export
#' @family nn ops
#' @family ops
#' @tether keras.ops.batch_normalization
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/batch_normalization>
op_batch_normalization <-
function (x, mean, variance, axis, offset = NULL, scale = NULL,
    epsilon = 0.001)
{
    args <- capture_args(list(
      axis = as_axis,
      mean = as_array,
      variance = as_array,
      offset = as_array
    ))
    do.call(keras$ops$batch_normalization, args)
}

#' Normalizes `x` over the specified axis.
#'
#' @description
#' It is defined as: `op_normalize(x) = x / max(norm(x), epsilon)`.
#'
#' # Examples
#' ```{r}
#' x <- op_convert_to_tensor(rbind(c(1, 2, 3), c(4, 5, 6)))
#' x_norm <- op_normalize(x)
#' x_norm
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
#' @param epsilon
#' A lower bound value for the norm.
#' Defaults to `config_epsilon()`.
#'
#' @export
#' @family nn ops
#' @family ops
#' @tether keras.ops.normalize
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/normalize>
op_normalize <-
function (x, axis = -1L, order = 2L, epsilon = NULL)
{
    args <- capture_args(list(axis = as_axis, order = as_integer))
    do.call(keras$ops$normalize, args)
}


#' Peak Signal-to-Noise Ratio (PSNR) function.
#'
#' @description
#' This function computes the Peak Signal-to-Noise Ratio between two signals,
#' `x1` and `x2`. PSNR is a measure of the quality of a reconstructed signal.
#' The higher the PSNR, the closer the reconstructed signal is to the original
#' signal. Note that it can become negative when the signal power is
#' smaller that the noise power.
#'
#' # Examples
#' ```{r}
#' x1 <- random_normal(c(2, 4, 4, 3))
#' x2 <- random_normal(c(2, 4, 4, 3))
#' max_val <- 1.0
#' op_psnr(x1, x2, max_val)
#' ```
#'
#' @returns
#' float: The PSNR value between `x1` and `x2`.
#'
#' @param x1
#' The first input signal.
#'
#' @param x2
#' The second input signal. Must have the same shape as `x1`.
#'
#' @param max_val
#' The maximum possible value in the signals.
#'
#' @export
#' @family nn ops
#' @family ops
#' @tether keras.ops.psnr
op_psnr <-
function (x1, x2, max_val)
keras$ops$psnr(x1, x2, max_val)



#' Scaled dot product attention function.
#'
#' @description
#' Computes the attention function on Q (`query`), K (`key`), and V(`value`):
#' `attention(Q, K, V) = softmax(Q * K / sqrt(d)) * V`. If we define `logits`
#' as the output of `Q * K` and the `probs` as the output of `softmax`.
#'
#' Throughout this function, we utilize the following notation to represent the
#' shape of array:
#' - B: batch size
#' - S: length of the key/value
#' - T: length of the query
#' - N: number of attention heads
#' - H: dimensions of each attention head
#' - K: number of key/value heads
#' - G: number of groups, which equals to `N // K`
#'
#' # Examples
#' ```{r}
#' query = random_normal(c(2, 4, 8, 16))
#' key = random_normal(c(2, 6, 8, 16))
#' value = random_normal(c(2, 6, 8, 16))
#' op_dot_product_attention(query, key, value) |> op_shape()
#' ```
#'
#' @returns
#' An array of the attention output with the same shape of `query`.
#'
#' @param query
#' The query array with the shape of `(B, T, N, H)`.
#'
#' @param key
#' The key array with the shape of `(B, S, K, H)`. When `K` equals
#' `N`, multi-headed attention (MHA) is performed. Otherwise, grouped
#' query attention (GQA) is performed if `N` is a multiple of `K`. and
#' multi-query attention (MQA) is performed if `K==1` (a special case
#' of GQA).
#'
#' @param value
#' The value array with the same shape of `key`.
#'
#' @param bias
#' Optional bias array to be added to logits. The shape must be
#' broadcastable to `(B, N, T, S)`.
#'
#' @param mask
#' Optional mask array used to filter out logits. It is a boolean
#' mask where `TRUE` indicates the element should take part in
#' attention. For an additive mask, users should pass it to bias. The
#' shape must be broadcastable to `(B, N, T, S)`.
#'
#' @param scale
#' Optional scale for the logits. If `NULL`, the scale will be set
#' to `1.0 / sqrt(H)`.
#'
#' @param is_causal
#' Whether to apply causal mask.
#'
#' @export
#' @tether keras.ops.dot_product_attention
#' @family nn ops
#' @family ops
op_dot_product_attention <-
function (query, key, value, bias = NULL, mask = NULL, scale = NULL,
          is_causal = FALSE)
{
  args <- capture_args()
  do.call(keras$ops$dot_product_attention, args)
}
