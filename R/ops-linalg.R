#' Computes the Cholesky decomposition of a positive semi-definite matrix.
#'
#' @returns
#' A tensor of shape `(..., M, M)` representing the lower triangular
#' Cholesky factor of `x`.
#'
#' @param x
#' Input tensor of shape `(..., M, M)`.
#'
#' @export
#' @tether keras.ops.cholesky
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/cholesky>
op_cholesky <-
function (x)
keras$ops$cholesky(x)

#' Computes the determinant of a square tensor.
#'
#' @returns
#'     A tensor of shape `(...,)` represeting the determinant of `x`.
#'
#' @param x
#' Input tensor of shape `(..., M, M)`.
#'
#' @export
#' @tether keras.ops.det
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/det>
op_det <-
function (x)
keras$ops$det(x)

#' Computes the eigenvalues and eigenvectors of a square matrix.
#'
#' @returns
#' A tuple of two tensors: a tensor of shape `(..., M)` containing
#' eigenvalues and a tensor of shape `(..., M, M)` containing eigenvectors.
#'
#' @param x
#' Input tensor of shape `(..., M, M)`.
#'
#' @export
#' @tether keras.ops.eig
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/eig>
op_eig <-
function (x)
keras$ops$eig(x)

#' Computes the inverse of a square tensor.
#'
#' @returns
#'     A tensor of shape `(..., M, M)` representing the inverse of `x`.
#'
#' @param x
#' Input tensor of shape `(..., M, M)`.
#'
#' @export
#' @tether keras.ops.inv
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/inv>
op_inv <-
function (x)
keras$ops$inv(x)

#' Computes the lower-upper decomposition of a square matrix.
#'
#' @returns
#' A tuple of two tensors: a tensor of shape `(..., M, M)` containing the
#' lower and upper triangular matrices and a tensor of shape `(..., M)`
#' containing the pivots.
#'
#' @param x
#' A tensor of shape `(..., M, M)`.
#'
#' @export
#' @tether keras.ops.lu_factor
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/lu_factor>
op_lu_factor <-
function (x)
keras$ops$lu_factor(x)

#' Matrix or vector norm.
#'
#' @description
#' This function is able to return one of eight different matrix norms, or one
#' of an infinite number of vector norms (described below), depending on the
#' value of the `ord` parameter.
#'
#' # Note
#' For values of `ord < 1`, the result is, strictly speaking, not a
#' mathematical 'norm', but it may still be useful for various numerical
#' purposes. The following norms can be calculated:
#' - For matrices:
#'     - `ord=None`: Frobenius norm
#'     - `ord="fro"`: Frobenius norm
#'     - `ord=nuc`: nuclear norm
#'     - `ord=np.inf`: `max(sum(abs(x), axis=1))`
#'     - `ord=-np.inf`: `min(sum(abs(x), axis=1))`
#'     - `ord=0`: not supported
#'     - `ord=1`: `max(sum(abs(x), axis=0))`
#'     - `ord=-1`: `min(sum(abs(x), axis=0))`
#'     - `ord=2`: 2-norm (largest sing. value)
#'     - `ord=-2`: smallest singular value
#'     - other: not supported
#' - For vectors:
#'     - `ord=None`: 2-norm
#'     - `ord="fro"`: not supported
#'     - `ord=nuc`: not supported
#'     - `ord=np.inf`: `max(abs(x))`
#'     - `ord=-np.inf`: `min(abs(x))`
#'     - `ord=0`: `sum(x != 0)`
#'     - `ord=1`: as below
#'     - `ord=-1`: as below
#'     - `ord=2`: as below
#'     - `ord=-2`: as below
#'     - other: `sum(abs(x)**ord)**(1./ord)`
#'
#' # Examples
#' ```python
#' x = keras.ops.reshape(keras.ops.arange(9, dtype="float32") - 4, (3, 3))
#' keras.ops.linalg.norm(x)
#' # 7.7459664
#' ```
#'
#' @returns
#' Norm of the matrix or vector(s).
#'
#' @param x
#' Input tensor.
#'
#' @param ord
#' Order of the norm (see table under Notes). The default is `None`.
#'
#' @param axis
#' If `axis` is an integer, it specifies the axis of `x` along which
#' to compute the vector norms. If `axis` is a 2-tuple, it specifies
#' the axes that hold 2-D matrices, and the matrix norms of these
#' matrices are computed.
#'
#' @param keepdims
#' If this is set to `True`, the axes which are reduced are left
#' in the result as dimensions with size one.
#'
#' @export
#' @tether keras.ops.norm
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/norm>
op_norm <-
function (x, ord = NULL, axis = NULL, keepdims = FALSE)
{
    args <- capture_args(list(axis = as_axis))
    do.call(keras$ops$norm, args)
}

#' Solves a linear system of equations given by `a x = b`.
#'
#' @returns
#' A tensor of shape `(..., M)` or `(..., M, N)` representing the solution
#' of the linear system. Returned shape is identical to `b`.
#'
#' @param a
#' A tensor of shape `(..., M, M)` representing the coefficients matrix.
#'
#' @param b
#' A tensor of shape `(..., M)` or `(..., M, N)` represeting the
#' right-hand side or "dependent variable" matrix.
#'
#' @export
#' @tether keras.ops.solve_triangular
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/solve_triangular>
op_solve_triangular <-
function (a, b, lower = FALSE)
keras$ops$solve_triangular(a, b, lower)

#' Computes the singular value decomposition of a matrix.
#'
#' @returns
#' A tuple of three tensors: a tensor of shape `(..., M, M)` containing the
#' left singular vectors, a tensor of shape `(..., M, N)` containing the
#' singular values and a tensor of shape `(..., N, N)` containing the
#' right singular vectors.
#'
#' @param x
#' Input tensor of shape `(..., M, N)`.
#'
#' @export
#' @tether keras.ops.svd
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/svd>
op_svd <-
function (x, full_matrices = TRUE, compute_uv = TRUE)
keras$ops$svd(x, full_matrices, compute_uv)