#' Return a tensor with ones at and below a diagonal and zeros elsewhere.
#'
#' @description
#'
#' # Returns
#' Tensor with its lower triangle filled with ones and zeros elsewhere.
#' `T[i, j] == 1` for `j <= i + k`, 0 otherwise.
#'
#' @param N Number of rows in the tensor.
#' @param M Number of columns in the tensor.
#' @param k The sub-diagonal at and below which the array is filled.
#'     `k = 0` is the main diagonal, while `k < 0` is below it, and
#'     `k > 0` is above. The default is 0.
#' @param dtype Data type of the returned tensor. The default is "float32".
#'
#' @export
#' @family ops
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/tri>
k_tri <-
function (N, M = NULL, k = 0L, dtype = NULL)
{
    args <- capture_args2(list(k = as_integer))
    do.call(keras$ops$tri, args)
}
