
#' L1 and L2 regularization
#'
#' @param l Regularization factor.
#' @param l1 L1 regularization factor.
#' @param l2 L2 regularization factor.
#'
#' @export
regularizer_l1 <- function(l = 0.01) {
  keras$regularizers$l1(l = l)
}

#' @rdname regularizer_l1
#' @export
regularizer_l2 <- function(l = 0.01) {
  keras$regularizers$l2(l = l)
}

#' @rdname regularizer_l1
#' @export
regularizer_l1_l2 <- function(l1 = 0.01, l2 = 0.01) {
  keras$regularizers$l1_l2(l1 = l1, l2 = l2)
}


#' A regularizer that encourages input vectors to be orthogonal to each other
#'
#' @details
#' It can be applied to either the rows of a matrix (`mode="rows"`) or its
#' columns (`mode="columns"`). When applied to a `Dense` kernel of shape
#' `(input_dim, units)`, rows mode will seek to make the feature vectors
#' (i.e. the basis of the output space) orthogonal to each other.
#'
#' @param factor Float. The regularization factor. The regularization penalty will
#' be proportional to `factor` times the mean of the dot products between
#' the L2-normalized rows (if `mode="rows"`, or columns if `mode="columns"`)
#' of the inputs, excluding the product of each row/column with itself.
#' Defaults to 0.01.
#'
#' @param mode String, one of `{"rows", "columns"}`. Defaults to `"rows"`. In rows
#' mode, the regularization effect seeks to make the rows of the input
#' orthogonal to each other. In columns mode, it seeks to make the columns
#' of the input orthogonal to each other.
#' @param ... For backwards and forwards compatibility
#'
#' ````r
#' layer <- layer_dense(
#'   units = 4,
#'   kernel_regularizer = regularizer_orthogonal(factor = 0.01))
#' ````
#'
#' @seealso
#'   +  <https://www.tensorflow.org/api_docs/python/tf/keras/regularizers/OrthogonalRegularizer>
#' @export
regularizer_orthogonal <-
function(factor = 0.01, mode = "rows", ...)
{
  args <- capture_args(match.call(), NULL)
  do.call(keras$regularizers$OrthogonalRegularizer, args)
}
