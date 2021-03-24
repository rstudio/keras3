

#' Apply additive zero-centered Gaussian noise.
#' 
#' This is useful to mitigate overfitting (you could see it as a form of random
#' data augmentation). Gaussian Noise (GS) is a natural choice as corruption
#' process for real valued inputs. As it is a regularization layer, it is only
#' active at training time.
#' 
#' @inheritParams layer_dense
#' 
#' @param stddev float, standard deviation of the noise distribution.
#'   
#' @section Input shape: Arbitrary. Use the keyword argument `input_shape` (list
#'   of integers, does not include the samples axis) when using this layer as
#'   the first layer in a model.
#'   
#' @section Output shape: Same shape as input.
#'   
#' @family noise layers   
#'   
#' @export
layer_gaussian_noise <- function(object, stddev, input_shape = NULL,
                                 batch_input_shape = NULL, batch_size = NULL, dtype = NULL, 
                                 name = NULL, trainable = NULL, weights = NULL) {
  create_layer(keras$layers$GaussianNoise, object, list(
    stddev = stddev,
    input_shape = normalize_shape(input_shape),
    batch_input_shape = normalize_shape(batch_input_shape),
    batch_size = as_nullable_integer(batch_size),
    dtype = dtype,
    name = name,
    trainable = trainable,
    weights = weights
  ))
}

#' Apply multiplicative 1-centered Gaussian noise.
#' 
#' As it is a regularization layer, it is only active at training time.
#' 
#' @inheritParams layer_dense
#' 
#' @param rate float, drop probability (as with `Dropout`). The multiplicative
#'   noise will have standard deviation `sqrt(rate / (1 - rate))`.
#'   
#' @section Input shape: Arbitrary. Use the keyword argument `input_shape` (list
#'   of integers, does not include the samples axis) when using this layer as
#'   the first layer in a model.
#'   
#' @section Output shape: Same shape as input.
#'   
#' @section References: 
#' - [Dropout: A Simple Way to Prevent Neural Networks from Overfitting Srivastava, Hinton, et al. 2014](https://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)
#'   
#' @family noise layers   
#'   
#' @export
layer_gaussian_dropout <- function(object, rate, input_shape = NULL,
                                   batch_input_shape = NULL, batch_size = NULL, dtype = NULL, 
                                   name = NULL, trainable = NULL, weights = NULL) {
  create_layer(keras$layers$GaussianDropout, object, list(
    rate = rate,
    input_shape = normalize_shape(input_shape),
    batch_input_shape = normalize_shape(batch_input_shape),
    batch_size = as_nullable_integer(batch_size),
    dtype = dtype,
    name = name,
    trainable = trainable,
    weights = weights
  ))
}


#' Applies Alpha Dropout to the input.
#'
#' Alpha Dropout is a dropout that keeps mean and variance of inputs to their
#' original values, in order to ensure the self-normalizing property even after
#' this dropout.
#'
#' Alpha Dropout fits well to Scaled Exponential Linear Units by randomly
#' setting activations to the negative saturation value.
#'
#' @inheritParams layer_dense
#'
#' @param rate float, drop probability (as with `layer_dropout()`). The
#'   multiplicative noise will have standard deviation `sqrt(rate / (1 -
#'   rate))`.
#' @param noise_shape Noise shape
#' @param seed An integer to use as random seed.
#'
#' @section Input shape: Arbitrary. Use the keyword argument `input_shape` (list
#'   of integers, does not include the samples axis) when using this layer as
#'   the first layer in a model.
#'   
#' @section Output shape: Same shape as input.
#'   
#' @section References: 
#'   - [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)
#'   
#' @family noise layers   
#'
#' @export
layer_alpha_dropout <- function(object, rate, noise_shape = NULL, seed = NULL, input_shape = NULL,
                                batch_input_shape = NULL, batch_size = NULL, dtype = NULL, 
                                name = NULL, trainable = NULL, weights = NULL) {
  create_layer(keras$layers$AlphaDropout, object, list(
    rate = rate,
    noise_shape = noise_shape,
    seed = as_nullable_integer(seed),
    input_shape = normalize_shape(input_shape),
    batch_input_shape = normalize_shape(batch_input_shape),
    batch_size = as_nullable_integer(batch_size),
    dtype = dtype,
    name = name,
    trainable = trainable,
    weights = weights
  ))
}


