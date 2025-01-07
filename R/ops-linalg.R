
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
#' @family linear algebra ops
#' @family ops
#' @tether keras.ops.cholesky
# @seealso
# + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/cholesky>
op_cholesky <-
function (x)
keras$ops$cholesky(x)


#' Computes the determinant of a square tensor.
#'
#' @returns
#' A tensor of shape `(...)` representing the determinant of `x`.
#'
#' @param x
#' Input tensor of shape `(..., M, M)`.
#'
#' @export
#' @family linear algebra ops
#' @family ops
#' @tether keras.ops.det
# @seealso
# + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/det>
op_det <-
function (x)
keras$ops$det(x)


#' Computes the eigenvalues and eigenvectors of a square matrix.
#'
#' @returns
#' A list of two tensors: a tensor of shape `(..., M)` containing
#' eigenvalues and a tensor of shape `(..., M, M)` containing eigenvectors.
#'
#' @param x
#' Input tensor of shape `(..., M, M)`.
#'
#' @export
#' @family linear algebra ops
#' @family ops
#' @tether keras.ops.eig
# @seealso
# + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/eig>
op_eig <-
function (x)
keras$ops$eig(x)

#' Computes the eigenvalues and eigenvectors of a complex Hermitian.
#'
#' @returns
#' A list of two tensors: a tensor of shape `(..., M)` containing
#' eigenvalues and a tensor of shape `(..., M, M)` containing eigenvectors.
#'
#' @param x
#' Input tensor of shape `(..., M, M)`.
#'
#' @export
#' @family linear algebra ops
#' @family ops
#' @tether keras.ops.eigh
op_eigh <-
function (x)
keras$ops$eigh(x)

#' Computes the inverse of a square tensor.
#'
#' @returns
#' A tensor of shape `(..., M, M)` representing the inverse of `x`.
#'
#' @param x
#' Input tensor of shape `(..., M, M)`.
#'
#' @export
#' @family linear algebra ops
#' @family ops
#' @tether keras.ops.inv
# @seealso
# + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/inv>
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
#' @family linear algebra ops
#' @family ops
#' @tether keras.ops.lu_factor
# @seealso
# + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/lu_factor>
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
#'     - `ord=NULL`: Frobenius norm
#'     - `ord="fro"`: Frobenius norm
#'     - `ord="nuc"`: nuclear norm
#'     - `ord=Inf`: `max(sum(abs(x), axis=2))`
#'     - `ord=-Inf`: `min(sum(abs(x), axis=2))`
#'     - `ord=0`: not supported
#'     - `ord=1`: `max(sum(abs(x), axis=1))`
#'     - `ord=-1`: `min(sum(abs(x), axis=1))`
#'     - `ord=2`: 2-norm (largest sing. value)
#'     - `ord=-2`: smallest singular value
#'     - other: not supported
#' - For vectors:
#'     - `ord=NULL`: 2-norm
#'     - `ord="fro"`: not supported
#'     - `ord="nuc"`: not supported
#'     - `ord=Inf`: `max(abs(x))`
#'     - `ord=-Inf`: `min(abs(x))`
#'     - `ord=0`: `sum(x != 0)`
#'     - `ord=1`: as below
#'     - `ord=-1`: as below
#'     - `ord=2`: as below
#'     - `ord=-2`: as below
#'     - other: `sum(abs(x)^ord)^(1/ord)`
#'
#' # Examples
#' ```{r}
#' x <- op_reshape(op_arange(9, dtype="float32") - 4, c(3, 3))
#' op_norm(x)
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
#' Order of the norm (see table under Notes). The default is `NULL`.
#'
#' @param axis
#' If `axis` is an integer, it specifies the axis of `x` along which
#' to compute the vector norms. If `axis` is a length 2 vector, it specifies
#' the axes that hold 2-D matrices, and the matrix norms of these
#' matrices are computed.
#'
#' @param keepdims
#' If this is set to `TRUE`, the axes which are reduced are left
#' in the result as dimensions with size one.
#'
#' @export
#' @family linear algebra ops
#' @family ops
#' @tether keras.ops.norm
# @seealso
# + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/norm>
op_norm <-
function (x, ord = NULL, axis = NULL, keepdims = FALSE)
{
    args <- capture_args(list(
      axis = as_axis,
      ord = function(x) {
        if (is.double(x) && all(!is.infinite(x)))
          as.integer(x)
        else
          x
      }
    ))
    do.call(keras$ops$norm, args)
}


#' Solves a linear system of equations given by `a %*% x = b`.
#'
#' @returns
#' A tensor of shape `(..., M)` or `(..., M, N)` representing the solution
#' of the linear system. Returned shape is identical to `b`.
#'
#' @param a
#' A tensor of shape `(..., M, M)` representing the coefficients matrix.
#'
#' @param b
#' A tensor of shape `(..., M)` or `(..., M, N)` representing the
#' right-hand side or "dependent variable" matrix.
#'
#' @param lower logical.
#' Use only data contained in the lower triangle of `a`. Default is to use upper triangle.
#'
#' @export
#' @family linear algebra ops
#' @family ops
#' @tether keras.ops.solve_triangular
# @seealso
# + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/solve_triangular>
op_solve_triangular <-
function (a, b, lower = FALSE)
keras$ops$solve_triangular(a, b, lower)


#' Computes the singular value decomposition of a matrix.
#'
#' @returns
#' A list of three tensors:
#' - a tensor of shape `(..., M, M)` containing the
#'   left singular vectors,
#' - a tensor of shape `(..., M, N)` containing the
#'   singular values and
#' - a tensor of shape `(..., N, N)` containing the
#'   right singular vectors.
#'
#' @param x
#' Input tensor of shape `(..., M, N)`.
#'
#' @param full_matrices Logical
#' @param compute_uv Logical
#'
#' @export
#' @family linear algebra ops
#' @family ops
#' @tether keras.ops.svd
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/svd>
op_svd <-
function (x, full_matrices = TRUE, compute_uv = TRUE)
keras$ops$svd(x, full_matrices, compute_uv)



#' Compute the sign and natural logarithm of the determinant of a matrix.
#'
#' @returns
#' A list: `(sign, logabsdet)`. `sign` is a number representing
#' the sign of the determinant. For a real matrix, this is 1, 0, or -1.
#' For a complex matrix, this is a complex number with absolute value 1
#' (i.e., it is on the unit circle), or else 0.
#' `logabsdet` is the natural log of the absolute value of the determinant.
#'
#' @param x
#' Input matrix. It must 2D and square.
#'
#' @export
#' @family linear algebra ops
#' @family ops
#' @tether keras.ops.slogdet
op_slogdet <-
function (x)
keras$ops$slogdet(x)


#' Return the least-squares solution to a linear matrix equation.
#'
#' @description
#' Computes the vector x that approximately solves the equation
#' `a %*% x = b`. The equation may be under-, well-, or over-determined
#' (i.e., the number of linearly independent rows of a can be less than,
#' equal to, or greater than its number of linearly independent columns).
#' If a is square and of full rank, then `x` (but for round-off error)
#' is the exact solution of the equation. Else, `x` minimizes the
#' L2 norm of `b - a %*% x`.
#'
#' If there are multiple minimizing solutions,
#' the one with the smallest L2 norm  is returned.
#'
#' @returns
#' Tensor with shape `(N)` or `(N, K)` containing
#' the least-squares solutions.
#'
#' **NOTE:** The output differs from `numpy.linalg.lstsq()`.
#' NumPy returns a tuple with four elements, the first of which
#' being the least-squares solutions and the others
#' being essentially never used.
#' Keras only returns the first value. This is done both
#' to ensure consistency across backends (which cannot be achieved
#' for the other values) and to simplify the API.
#'
#' @param a
#' "Coefficient" matrix of shape `(M, N)`.
#'
#' @param b
#' Ordinate or "dependent variable" values,
#' of shape `(M)` or `(M, K)`.
#' If `b` is two-dimensional, the least-squares solution
#' is calculated for each of the K columns of `b`.
#'
#' @param rcond
#' Cut-off ratio for small singular values of `a`.
#' For the purposes of rank determination,
#' singular values are treated as zero if they are
#' smaller than rcond times the largest
#' singular value of `a`.
#'
#'
#' @family linear algebra ops
#' @family numpy ops
#' @family ops
#' @export
#' @tether keras.ops.lstsq
op_lstsq <-
function (a, b, rcond = NULL)
keras$ops$lstsq(a, b, rcond)
