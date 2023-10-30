#' Computes the Fast Fourier Transform along last axis of input.
#'
#' @description
#' Returns a tuple containing two tensors - the real and imaginary parts of the
#' output tensor.
#'
#' @param x Tuple of the real and imaginary parts of the input tensor. Both
#' tensors in the tuple should be of floating type.
#' @export
#' @family ops
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/fft>
#'
#' @examples
#' x <- list(
#'     tf$convert_to_tensor(c(1., 2.)),
#'     tf$convert_to_tensor(c(0., 1.))
#' )
#' fft(x)
#' # list(c(3., -1.), c(1., -1.))
k_fft <-
function (x)
keras$ops$fft(x)
