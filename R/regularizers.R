


#' A regularizer that applies a L1 regularization penalty.
#'
#' @description
#' The L1 regularization penalty is computed as:
#' `loss = l1 * reduce_sum(abs(x))`
#'
#' L1 may be passed to a layer as a string identifier:
#'
#' ```{r}
#' dense <- layer_dense(units = 3, kernel_regularizer = 'l1')
#' ```
#'
#' In this case, the default value used is `l1=0.01`.
#'
#' @param l1
#' float, L1 regularization factor.
#'
#' @export
#' @family regularizers
#' @seealso
#' + <https:/keras.io/api/layers/regularizers#l1-class>
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/regularizers/L1>
#'
#' @tether keras.regularizers.L1
regularizer_l1 <-
function (l1 = 0.01)
{
    args <- capture_args()
    do.call(keras$regularizers$L1, args)
}


#' A regularizer that applies both L1 and L2 regularization penalties.
#'
#' @description
#' The L1 regularization penalty is computed as:
#' `loss = l1 * reduce_sum(abs(x))`
#'
#' The L2 regularization penalty is computed as
#' `loss = l2 * reduce_sum(square(x))`
#'
#' L1L2 may be passed to a layer as a string identifier:
#'
#' ```{r}
#' dense <- layer_dense(units = 3, kernel_regularizer = 'L1L2')
#' ```
#'
#' In this case, the default values used are `l1=0.01` and `l2=0.01`.
#'
#' @param l1
#' float, L1 regularization factor.
#'
#' @param l2
#' float, L2 regularization factor.
#'
#' @export
#' @family regularizers
#' @seealso
#' + <https:/keras.io/api/layers/regularizers#l1l2-class>
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/regularizers/L1L2>
#'
#' @tether keras.regularizers.L1L2
regularizer_l1_l2 <-
function (l1 = 0, l2 = 0)
{
    args <- capture_args()
    do.call(keras$regularizers$L1L2, args)
}


#' A regularizer that applies a L2 regularization penalty.
#'
#' @description
#' The L2 regularization penalty is computed as:
#' `loss = l2 * reduce_sum(square(x))`
#'
#' L2 may be passed to a layer as a string identifier:
#'
#' ```{r}
#' dense <- layer_dense(units = 3, kernel_regularizer='l2')
#' ```
#'
#' In this case, the default value used is `l2=0.01`.
#'
#' @param l2
#' float, L2 regularization factor.
#'
#' @export
#' @family regularizers
#' @seealso
#' + <https:/keras.io/api/layers/regularizers#l2-class>
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/regularizers/L2>
#'
#' @tether keras.regularizers.L2
regularizer_l2 <-
function (l2 = 0.01)
{
    args <- capture_args()
    do.call(keras$regularizers$L2, args)
}


#' Regularizer that encourages input vectors to be orthogonal to each other.
#'
#' @description
#' It can be applied to either the rows of a matrix (`mode="rows"`) or its
#' columns (`mode="columns"`). When applied to a `Dense` kernel of shape
#' `(input_dim, units)`, rows mode will seek to make the feature vectors
#' (i.e. the basis of the output space) orthogonal to each other.
#'
#' # Examples
#' ```{r}
#' regularizer <- regularizer_orthogonal(factor=0.01)
#' layer <- layer_dense(units=4, kernel_regularizer=regularizer)
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
#'
#' @tether keras.regularizers.OrthogonalRegularizer
regularizer_orthogonal <-
function (factor = 0.01, mode = "rows")
{
    args <- capture_args()
    do.call(keras$regularizers$OrthogonalRegularizer, args)
}
