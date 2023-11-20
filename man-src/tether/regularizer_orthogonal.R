#' Regularizer that encourages input vectors to be orthogonal to each other.
#'
#' @description
#' It can be applied to either the rows of a matrix (`mode="rows"`) or its
#' columns (`mode="columns"`). When applied to a `Dense` kernel of shape
#' `(input_dim, units)`, rows mode will seek to make the feature vectors
#' (i.e. the basis of the output space) orthogonal to each other.
#'
#' # Examples
#' ```python
#' regularizer = OrthogonalRegularizer(factor=0.01)
#' layer = Dense(units=4, kernel_regularizer=regularizer)
#' ```
#'
#' @param factor
#' Float. The regularization factor. The regularization penalty
#' will be proportional to `factor` times the mean of the dot products
#' between the L2-normalized rows (if `mode="rows"`, or columns if
#' `mode="columns"`) of the inputs, excluding the product of each
#' row/column with itself.  Defaults to `0.01`.
#'
#' @param mode
#' String, one of `{"rows", "columns"}`. Defaults to `"rows"`. In
#' rows mode, the regularization effect seeks to make the rows of the
#' input orthogonal to each other. In columns mode, it seeks to make
#' the columns of the input orthogonal to each other.
#'
#' @export
#' @family regularizers
#' @seealso
#' + <https:/keras.io/api/layers/regularizers#orthogonalregularizer-class>
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/regularizers/OrthogonalRegularizer>
regularizer_orthogonal <-
function (factor = 0.01, mode = "rows")
{
}
