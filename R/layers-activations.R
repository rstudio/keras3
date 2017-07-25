
#' Apply an activation function to an output.
#' 
#' @inheritParams layer_dense
#'   
#' @param input_shape Input shape (list of integers, does not include the
#'   samples axis) which is required when using this layer as the first layer in
#'   a model.
#'   
#' @family core layers
#' @family activation layers    
#'   
#' @export
layer_activation <- function(object, activation, input_shape = NULL,
                             batch_input_shape = NULL, batch_size = NULL, dtype = NULL, 
                             name = NULL, trainable = NULL, weights = NULL) {
  
  create_layer(keras$layers$Activation, object, list(
    activation = activation,
    input_shape = normalize_shape(input_shape),
    batch_input_shape = normalize_shape(batch_input_shape),
    batch_size = as_nullable_integer(batch_size),
    dtype = dtype,
    name = name,
    trainable = trainable,
    weights = weights
  ))
  
}

#' Leaky version of a Rectified Linear Unit.
#'
#' Allows a small gradient when the unit is not active: `f(x) = alpha * x` for
#' `x < 0`, `f(x) = x` for `x >= 0`.
#'
#' @inheritParams layer_activation
#' @param alpha float >= 0. Negative slope coefficient.
#'
#' @seealso [Rectifier Nonlinearities Improve Neural Network Acoustic
#'   Models](https://ai.stanford.edu/~amaas/papers/relu_hybrid_icml2013_final.pdf).
#'
#' @family activation layers
#'
#' @export
layer_activation_leaky_relu <- function(object, alpha = 0.3, input_shape = NULL,
                                        batch_input_shape = NULL, batch_size = NULL, 
                                        dtype = NULL, name = NULL, trainable = NULL, 
                                        weights = NULL) {
  
  create_layer(keras$layers$LeakyReLU, object, list(
    alpha = alpha,
    input_shape = normalize_shape(input_shape),
    batch_input_shape = normalize_shape(batch_input_shape),
    batch_size = as_nullable_integer(batch_size),
    dtype = dtype,
    name = name,
    trainable = trainable,
    weights = weights
  ))
  
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
#' @seealso [Delving Deep into Rectifiers: Surpassing Human-Level Performance on
#'   ImageNet Classification](https://arxiv.org/abs/1502.01852).
#'
#' @family activation layers
#'
#' @export
layer_activation_parametric_relu <- function(object, alpha_initializer = "zeros", alpha_regularizer = NULL, 
                                             alpha_constraint = NULL, shared_axes = NULL, 
                                             input_shape = NULL,
                                             batch_input_shape = NULL, batch_size = NULL, 
                                             dtype = NULL, name = NULL, trainable = NULL, 
                                             weights = NULL) {
  
  # build args
  args <- list(
    alpha_initializer = alpha_initializer,
    alpha_regularizer = alpha_regularizer,
    alpha_constraint = alpha_constraint
  )
  if (!is.null(shared_axes))
    args$shared_axes <- as.list(as.integer(shared_axes))
  args$input_shape <- normalize_shape(input_shape)
  args$batch_input_shape <- normalize_shape(batch_input_shape)
  args$batch_size <- as_nullable_integer(batch_size)
  args$dtype <- dtype
  args$name <- name
  args$trainable <- trainable
  args$weights <- weights
  
  # call layer
  create_layer(keras$layers$PReLU, object, args)
}


#' Thresholded Rectified Linear Unit.
#'
#' It follows: `f(x) = x` for `x > theta`, `f(x) = 0` otherwise.
#'
#' @inheritParams layer_activation
#' @param theta float >= 0. Threshold location of activation.
#'
#' @seealso [Zero-bias autoencoders and the benefits of co-adapting features](https://arxiv.org/abs/1402.3337).
#'
#' @family activation layers
#'
#' @export
layer_activation_thresholded_relu <- function(object, theta = 1.0, input_shape = NULL,
                                              batch_input_shape = NULL, batch_size = NULL, 
                                              dtype = NULL, name = NULL, trainable = NULL, 
                                              weights = NULL) {
  
  create_layer(keras$layers$ThresholdedReLU, object, list(
    theta = theta,
    input_shape = normalize_shape(input_shape),
    batch_input_shape = normalize_shape(batch_input_shape),
    batch_size = as_nullable_integer(batch_size),
    dtype = dtype,
    name = name,
    trainable = trainable,
    weights = weights
  ))
  
}


#' Exponential Linear Unit.
#'
#' It follows: `f(x) =  alpha * (exp(x) - 1.0)` for `x < 0`, `f(x) = x` for `x
#' >= 0`.
#'
#' @inheritParams layer_activation
#' @param alpha Scale for the negative factor.
#'
#' @seealso [Fast and Accurate Deep Network Learning by Exponential Linear Units
#'   (ELUs)](https://arxiv.org/abs/1511.07289v1).
#'
#' @family activation layers
#'
#' @export
layer_activation_elu <- function(object, alpha = 1.0, input_shape = NULL,
                                 batch_input_shape = NULL, batch_size = NULL, dtype = NULL, 
                                 name = NULL, trainable = NULL, weights = NULL) {
  
  create_layer(keras$layers$ELU, object, list(
    alpha = alpha,
    input_shape = normalize_shape(input_shape),
    batch_input_shape = normalize_shape(batch_input_shape),
    batch_size = as_nullable_integer(batch_size),
    dtype = dtype,
    name = name,
    trainable = trainable,
    weights = weights
  ))
  
}

