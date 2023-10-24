#' Extract a diagonal or construct a diagonal array.
#'
#' @description
#'
#' # Returns
#' The extracted diagonal or constructed diagonal tensor.
#'
#' # Examples
#' ```python
#' from keras import ops
#' x = ops.arange(9).reshape((3, 3))
#' x
#' # array([[0, 1, 2],
#' #        [3, 4, 5],
#' #        [6, 7, 8]])
#' ```
#'
#' ```python
#' ops.diag(x)
#' # array([0, 4, 8])
#' ops.diag(x, k=1)
#' # array([1, 5])
#' ops.diag(x, k=-1)
#' # array([3, 7])
#' ```
#'
#' ```python
#' ops.diag(ops.diag(x)))
#' # array([[0, 0, 0],
#' #        [0, 4, 0],
#' #        [0, 0, 8]])
#' ```
#'
#' @param x Input tensor. If `x` is 2-D, returns the k-th diagonal of `x`.
#'     If `x` is 1-D, return a 2-D tensor with `x` on the k-th diagonal.
#' @param k The diagonal to consider. Defaults to `0`. Use `k > 0` for diagonals
#'     above the main diagonal, and `k < 0` for diagonals below
#'     the main diagonal.
#'
#' @export
#' @family ops
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/diag>
k_diag <-
function (x, k = 0L)
{
    args <- capture_args2(list(k = as_integer))
    do.call(keras$ops$diag, args)
}
