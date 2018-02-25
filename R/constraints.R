#' Weight constraints
#'
#' Functions that impose constraints on weight values.
#'
#' @details 
#'   - `constraint_maxnorm()` constrains the weights incident to each
#'      hidden unit to have a norm less than or equal to a desired value.
#'   - `constraint_nonneg()` constraints the weights to be non-negative
#'   - `constraint_unitnorm()` constrains the weights incident to each hidden
#'      unit to have unit norm.
#'   - `constraint_minmaxnorm()` constrains the weights incident to each 
#'      hidden unit to have the norm between a lower bound and an upper bound.
#'   
#' @param axis The axis along which to calculate weight norms. For instance, in
#'   a dense layer the weight matrix has shape `input_dim, output_dim`, set
#'   `axis` to `0` to constrain each weight vector of length `input_dim,`. In a
#'   convolution 2D layer with `dim_ordering="tf"`, the weight tensor has shape
#'   `rows, cols, input_depth, output_depth`, set `axis` to `c(0, 1, 2)` to
#'   constrain the weights of each filter tensor of size `rows, cols,
#'   input_depth`.
#' @param min_value The minimum norm for the incoming weights.
#' @param max_value The maximum norm for the incoming weights.
#' @param rate The rate for enforcing the constraint: weights will be rescaled to
#'   yield (1 - rate) * norm + rate * norm.clip(low, high). Effectively, this
#'   means that rate=1.0 stands for strict enforcement of the constraint, while
#'   rate<1.0 means that weights will be rescaled at each step to slowly move
#'   towards a value inside the desired interval.
#'
#'
#' @section Custom constraints:
#' 
#' You can implement your own constraint functions in R. A custom 
#' constraint is an R function that takes weights (`w`) as input
#' and returns modified weights. Note that keras [backend()] tensor
#' functions (e.g. [k_greater_equal()]) should be used in the 
#' implementation of custom constraints. For example:
#' 
#' ```r
#' nonneg_constraint <- function(w) {
#'   w * k_cast(k_greater_equal(w, 0), k_floatx())
#' }
#' 
#' layer_dense(units = 32, input_shape = c(784), 
#'             kernel_constraint = nonneg_constraint)
#' ```
#' 
#' Note that models which use custom constraints cannot be serialized using
#' [save_model_hdf5()]. Rather, the weights of the model should be saved
#' and restored using [save_model_weights_hdf5()].
#'
#' @seealso [Dropout: A Simple Way to Prevent Neural Networks from Overfitting
#'   Srivastava, Hinton, et al.
#'   2014](http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)
#'
#' @name constraints
#' 
#' @seealso [KerasConstraint]
#' 
#' @export
constraint_maxnorm <- function(max_value = 2, axis = 0) {
  keras$constraints$MaxNorm(max_value = as.integer(max_value), axis = as.integer(axis))
}


#' @rdname constraints
#' @export
constraint_nonneg <- function() {
  keras$constraints$NonNeg()
}


#' @rdname constraints
#' @export
constraint_unitnorm <- function(axis = 0) {
  keras$constraints$UnitNorm(axis = as.integer(axis))
}
  
#' @rdname constraints       
#' @export
constraint_minmaxnorm <- function(min_value = 0.0, max_value = 1.0, rate = 1.0, axis = 0) {
  keras$constraints$MinMaxNorm(min_value = min_value, max_value = max_value, rate = rate, axis = as.integer(axis))
}


#' Base R6 class for Keras constraints
#' 
#' @docType class
#' 
#' @format An [R6Class] generator object
#' 
#' @section Methods:
#' \describe{
#'  \item{\code{call(w)}}{Constrain the specified weights.}
#' }
#' 
#' @details You can implement a custom constraint either by creating an
#'  R function that accepts a weights (`w`) parameter, or by creating
#'  an R6 class that derives from `KerasConstraint` and implements a
#'  `call` method.
#'  
#' @note 
#' Models which use custom constraints cannot be serialized using 
#' [save_model_hdf5()]. Rather, the weights of the model should be saved
#' and restored using [save_model_weights_hdf5()].
#'
#' @examples \dontrun{
#' CustomNonNegConstraint <- R6::R6Class(
#'   "CustomNonNegConstraint",
#'   inherit = KerasConstraint,
#'   public = list(
#'     call = function(x) {
#'        w * k_cast(k_greater_equal(w, 0), k_floatx())
#'     }
#'   )
#' )
#' 
#' layer_dense(units = 32, input_shape = c(784), 
#'             kernel_constraint = CustomNonNegConstraint$new())
#' }
#' 
#' @seealso [constraints]
#' 
#' @export
KerasConstraint <- R6::R6Class("KerasConstraint",
  public = list(
    call = function(w) {
      stop("Keras custom constraints must implement the call function")
    },
    get_config = function() {
      reticulate::dict()
    })
)

as_constraint <- function(constraint) {
  
  # helper to create constraint
  create_constraint <- function(call, get_config = NULL) {
    if (is.null(get_config))
      get_config <- function() dict()
    python_path <- system.file("python", package = "keras")
    tools <- import_from_path("kerastools", path = python_path)
    tools$constraint$RConstraint(call, get_config)
  }
  
  if (is.function(constraint)) {
    create_constraint(constraint)  
  } else if (inherits(constraint, "KerasConstraint")) {
    create_constraint(constraint$call, constraint$get_config)
  } else {
    constraint
  }
}

  

