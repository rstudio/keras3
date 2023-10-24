#' Swish (or Silu) activation function.
#'
#' @description
#' It is defined as: `swish(x) = x * sigmoid(x)`.
#'
#' The Swish (or Silu) activation function is a smooth,
#' non-monotonic function that is unbounded above and
#' bounded below.
#'
#' # Reference
#' - [Ramachandran et al., 2017](https://arxiv.org/abs/1710.05941)
#'
#' @param x Input tensor.
#'
#' @export
#' @family activation functions
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/activations/silu>
activation_silu <-
structure(function (x)
{
    args <- capture_args2(NULL)
    do.call(keras$activations$silu, args)
}, py_function_name = "silu")
