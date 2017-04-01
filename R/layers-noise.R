

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
layer_gaussian_noise <- function(x, stddev, input_shape = NULL) {
  call_layer(tf$contrib$keras$layers$GaussianNoise, x, list(
    stddev = stddev,
    input_shape = normalize_shape(input_shape)
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
#' - [Dropout: A Simple Way to Prevent Neural Networks from Overfitting Srivastava, Hinton, et al. 2014](http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)
#'   
#' @family noise layers   
#'   
#' @export
layer_gaussian_dropout <- function(x, rate, input_shape = NULL) {
  call_layer(tf$contrib$keras$layers$GaussianDropout, x, list(
    rate = rate,
    input_shape = normalize_shape(input_shape)
  ))
}



