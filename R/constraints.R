#' MaxNorm weight constraint
#' 
#' Constrains the weights incident to each hidden unit to have a norm less than
#' or equal to a desired value.
#' 
#' @param max_value The maximum norm for the incoming weights.
#' @param axis The axis along which to calculate weight norms. For instance, in
#'   a dense layer the weight matrix has shape `input_dim, output_dim`, 
#'   set `axis` to `0` to constrain each weight vector of length `input_dim,`.
#'   In a convolution 2D layer with `dim_ordering="tf"`, the weight tensor has
#'   shape `rows, cols, input_depth, output_depth`, set `axis` to `c(0, 1, 2)` 
#'   to constrain the weights of each filter tensor of size `rows, cols,
#'   input_depth`.
#'   
#' @seealso [Dropout: A Simple Way to Prevent Neural Networks from Overfitting Srivastava, Hinton, et al. 2014](http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)
#'   
#' @family constraints   
#'   
#' @export
constraint_maxnorm <- function(max_value = 2, axis = 0) {
  keras$constraints$MaxNorm(max_value = as.integer(max_value), axis = as.integer(axis))
}


#' NonNeg weight constraint
#' 
#' Constrains the weights to be non-negative.
#'
#' @family constraints  
#'
#' @export
constraint_nonneg <- function() {
  keras$constraints$NonNeg()
}


#' UnitNorm weight constraint
#' 
#' Constrains the weights incident to each hidden unit to have unit norm.
#'
#' @inheritParams constraint_maxnorm
#'   
#' @family constraints  
#'   
#' @export
constraint_unitnorm <- function(axis = 0) {
  keras$constraints$UnitNorm(axis = as.integer(axis))
}

#' MinMaxNorm weight constraint
#' 
#' Constrains the weights incident to each hidden unit to have the norm between 
#' a lower bound and an upper bound.
#' 
#' @inheritParams constraint_maxnorm
#' @param min_value The minimum norm for the incoming weights.
#' @param max_value The maximum norm for the incoming weights.
#' @param rate The rate for enforcing the constraint: weights will be rescaled to
#'   yield (1 - rate) * norm + rate * norm.clip(low, high). Effectively, this
#'   means that rate=1.0 stands for strict enforcement of the constraint, while
#'   rate<1.0 means that weights will be rescaled at each step to slowly move
#'   towards a value inside the desired interval.
#' 
#' @family constraints  
#'       
#' @export
constraint_minmaxnorm <- function(min_value = 0.0, max_value = 1.0, rate = 1.0, axis = 0) {
  keras$constraints$MinMaxNorm(min_value = min_value, max_value = max_value, rate = rate, axis = as.integer(axis))
}



