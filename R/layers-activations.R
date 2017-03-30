
#' Apply an activation function to an output.
#' 
#' @inheritParams layer_dense
#'   
#' @param input_shape Input shape (list of integers, does not include the
#'   samples axis) which is required when using this layer as the first layer in
#'   a model.
#'   
#' @export
layer_activation <- function(x, activation, input_shape = NULL) {
  
  # build args
  args <- list(activation = resolve_keras_function(activation))
  args$input_shape <- normalize_shape(input_shape)
  
  # call function
  layer <- do.call(keras$layers$Activation, args)
  
  # compose
  compose_layer(x, layer)
}

#' Leaky version of a Rectified Linear Unit.
#' 
#' Allows a small gradient when the unit is not active: `f(x) = alpha * x` for
#' `x < 0`, `f(x) = x` for `x >= 0`.
#' 
#' @inheritParams layer_activation
#' @param alpha float >= 0. Negative slope coefficient.
#'   
#' @seealso 
#' \href{https://web.stanford.edu/~awni/papers/relu_hybrid_icml2013_final.pdf}{Rectifier
#' Nonlinearities Improve Neural Network Acoustic Models}.
#' 
#' @export
layer_activation_leaky_relu <- function(x, alpha = 0.3, input_shape = NULL) {
  
  # build args
  args <- list(alpha = alpha)
  args$input_shape <- normalize_shape(input_shape)
  
  # call function
  layer <- do.call(keras$layers$LeakyReLU, args)
  
  # compose
  compose_layer(x, layer)
}

#' Parametric Rectified Linear Unit.
#' 
#' It follows: `f(x) = alpha * x`` for `x < 0`, `f(x) = x` for `x >= 0`, where 
#' alpha is a learned array with the same shape as x.
#' 
#' @inheritParams layer_activation
#' @param alpha_initializer Initializer function for the weights.
#' @param alpha_regularizer Regularizer for the weights.
#' @param alpha_constraint Constraint for the weights.
#' @param shared_axes The axes along which to share learnable parameters for the
#'   activation function. For example, if the incoming feature maps are from a 
#'   2D convolution with output shape (batch, height, width, channels), and you 
#'   wish to share parameters across space so that each filter only has one set 
#'   of parameters, set shared_axes=c(1, 2).
#'   
#' @seealso \href{https://arxiv.org/abs/1502.01852}{Delving Deep into
#'   Rectifiers: Surpassing Human-Level Performance on ImageNet Classification}
#'   
#' @export
layer_activation_parametric_relu <- function(x, alpha_initializer = "zeros", alpha_regularizer = NULL, 
                                             alpha_constraint = NULL, shared_axes = NULL, 
                                             input_shape = NULL) {
  # build args
  args <- list(
    alpha_initializer = alpha_initializer,
    alpha_regularizer = alpha_regularizer,
    alpha_constraint = alpha_constraint
  )
  if (!is.null(shared_axes))
    args$shared_axes <- as.list(as.integer(shared_axes))
  args$input_shape <- normalize_shape(input_shape)
  
  # call function
  layer <- do.call(keras$layers$PReLU, args)
  
  # compose
  compose_layer(x, layer)
}


#' Thresholded Rectified Linear Unit.
#' 
#' It follows: `f(x) = x` for `x > theta`, `f(x) = 0` otherwise.
#' 
#' @inheritParams layer_activation
#' @param theta float >= 0. Threshold location of activation.
#'   
#' @seealso \href{https://arxiv.org/abs/1402.3337}{Zero-bias autoencoders and
#'   the benefits of co-adapting features}
#'   
#' @export
layer_activation_thresholded_relu <- function(x, theta = 1.0, input_shape = NULL) {
  
  # build args
  args <- list(
    theta = theta
  )
  args$input_shape <- normalize_shape(input_shape)
  
  # call function
  layer <- do.call(keras$layers$ThresholdedReLU, args)
  
  # compose
  compose_layer(x, layer)
}


#' Exponential Linear Unit.
#' 
#' It follows: `f(x) =  alpha * (exp(x) - 1.0)` for `x < 0`, `f(x) = x` for `x
#' >= 0`.
#' 
#' @inheritParams layer_activation
#' @param alpha Scale for the negative factor.
#'   
#' @seealso \href{https://arxiv.org/abs/1511.07289v1}{Fast and Accurate Deep
#' Network Learning by Exponential Linear Units (ELUs)}.
#' 
#' @export
layer_activation_elu <- function(x, alpha = 1.0, input_shape = NULL) {
  
  # build args
  args <- list(alpha = alpha)
  args$input_shape <- normalize_shape(input_shape)
  
  # call function
  layer <- do.call(keras$layers$ELU, args)
  
  # compose
  compose_layer(x, layer)
}

