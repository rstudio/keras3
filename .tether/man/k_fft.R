#' Computes the Fast Fourier Transform along last axis of input.
#'
#' @description
#'
#' # Examples
#' ```python
#' x = (
#'     keras.ops.convert_to_tensor([1., 2.]),
#'     keras.ops.convert_to_tensor([0., 1.]),
#' )
#' fft(x)
#' # (array([ 3., -1.], dtype=float32), array([ 1., -1.], dtype=float32))
#' ```
#'
#' @returns
#' A tuple containing two tensors - the real and imaginary parts of the
#' output tensor.
#'
#' @param x
#' Tuple of the real and imaginary parts of the input tensor. Both
#' tensors in the tuple should be of floating type.
#'
#' @export
#' @family math ops
#' @family ops
#' @seealso
#' + <https:/keras.io/keras_core/api/ops/fft#fft-function>
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/fft>
k_fft <-
function (x)
{
}
