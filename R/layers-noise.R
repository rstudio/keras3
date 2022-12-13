

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
#' @param seed Integer, optional random seed to enable deterministic behavior.
#'
#' @param ... standard layer arguments.
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
layer_gaussian_noise <-
function(object, stddev, seed = NULL, ...)
{
  args <- capture_args(match.call(),
    modifiers = c(standard_layer_arg_modifiers,
                  seed = as_nullable_integer),
    ignore = "object")
  create_layer(keras$layers$GaussianNoise, object, args)
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
#' @param seed Integer, optional random seed to enable deterministic behavior.
#'
#' @param ... standard layer arguments.
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
#' @seealso
#'   +  <https://www.tensorflow.org/api_docs/python/tf/keras/layers/GaussianDropout>
#'
#' @family noise layers
#'
#' @export
layer_gaussian_dropout <-
function(object, rate, seed = NULL, ...)
{
  args <- capture_args(match.call(),
    modifiers = c(standard_layer_arg_modifiers,
                  seed = as_nullable_integer),
    ignore = "object")
  create_layer(keras$layers$GaussianDropout, object, args)
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
#' @param ... standard layer arguments.
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
#' @seealso <https://www.tensorflow.org/api_docs/python/tf/keras/layers/AlphaDropout>
#'
#' @family noise layers
#'
#' @export
layer_alpha_dropout <-
function(object, rate, noise_shape = NULL, seed = NULL, ...) {
  args <- capture_args(match.call(),
    modifiers = list(
      seed = as_nullable_integer,
      input_shape = normalize_shape,
      batch_input_shape = normalize_shape,
      batch_size = as_nullable_integer
    ),
    ignore = "object"
  )
  create_layer(keras$layers$AlphaDropout, object, args)
}
