#' Return a 2-D tensor with ones on the diagonal and zeros elsewhere.
#'
#' @description
#'
#' # Returns
#'     Tensor with ones on the k-th diagonal and zeros elsewhere.
#'
#' @param N Number of rows in the output.
#' @param M Number of columns in the output. If `None`, defaults to `N`.
#' @param k Index of the diagonal: 0 (the default) refers to the main
#'     diagonal, a positive value refers to an upper diagonal,
#'     and a negative value to a lower diagonal.
#' @param dtype Data type of the returned tensor.
#'
#' @export
#' @family ops
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/eye>
k_eye <-
function (N, M = NULL, k = 0L, dtype = NULL)
{
    args <- capture_args2(list(k = as_integer))
    do.call(keras$ops$eye, args)
}
