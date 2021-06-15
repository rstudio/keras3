

#' Initializer that generates tensors initialized to 0.
#'
#' @family initializers
#'
#' @export
initializer_zeros <- function() {
  keras$initializers$Zeros()
}

#' Initializer that generates tensors initialized to 1.
#'
#' @family initializers
#'
#' @export
initializer_ones <- function() {
  keras$initializers$Ones()
}

#' Initializer that generates tensors initialized to a constant value.
#'
#' @param value float; the value of the generator tensors.
#'
#' @family initializers
#'
#' @export
initializer_constant <- function(value = 0) {
  keras$initializers$Constant(
    value = value
  )
}


#' Initializer that generates tensors with a normal distribution.
#'
#' @param mean Mean of the random values to generate.
#' @param stddev  Standard deviation of the random values to generate.
#' @param seed Integer used to seed the random generator.
#'
#' @family initializers
#'
#' @export
initializer_random_normal <- function(mean = 0.0, stddev = 0.05, seed = NULL) {
  keras$initializers$RandomNormal(
    mean = mean,
    stddev = stddev,
    seed = as_nullable_integer(seed)
  )
}

#' Initializer that generates tensors with a uniform distribution.
#'
#'
#' @param minval Lower bound of the range of random values to generate.
#' @param maxval Upper bound of the range of random values to generate. Defaults to 1 for float types.
#' @param seed seed
#'
#' @family initializers
#'
#' @export
initializer_random_uniform <- function(minval = -0.05, maxval = 0.05, seed = NULL) {
  keras$initializers$RandomUniform(
    minval = minval,
    maxval = maxval,
    seed = as_nullable_integer(seed)
  )
}


#' Initializer that generates a truncated normal distribution.
#'
#' These values are similar to values from an [initializer_random_normal()]
#' except that values more than two standard deviations from the mean
#' are discarded and re-drawn. This is the recommended initializer for
#' neural network weights and filters.
#'
#' @inheritParams initializer_random_normal
#'
#' @family initializers
#'
#' @export
initializer_truncated_normal <- function(mean = 0.0, stddev = 0.05, seed = NULL) {
  keras$initializers$TruncatedNormal(
    mean = mean,
    stddev = stddev,
    seed = as_nullable_integer(seed)
  )
}

#' Initializer capable of adapting its scale to the shape of weights.
#'
#' With `distribution="normal"`, samples are drawn from a truncated normal
#' distribution centered on zero, with `stddev = sqrt(scale / n)` where n is:
#' - number of input units in the weight tensor, if mode = "fan_in"
#' - number of output units, if mode = "fan_out"
#' - average of the numbers of input and output units, if mode = "fan_avg"
#'
#' With `distribution="uniform"`, samples are drawn from a uniform distribution
#' within `-limit, limit`, with `limit = sqrt(3 * scale / n)`.
#'
#' @inheritParams initializer_random_normal
#'
#' @param scale Scaling factor (positive float).
#' @param mode One of "fan_in", "fan_out", "fan_avg".
#' @param distribution One of "truncated_normal", "untruncated_normal" and "uniform".
#'   For backward compatibility, "normal" will be accepted and converted to
#'   "untruncated_normal".
#'
#' @family initializers
#'
#' @export
initializer_variance_scaling <- function(scale = 1.0, mode = c("fan_in", "fan_out", "fan_avg"),
                                         distribution = c("normal", "uniform", "truncated_normal", "untruncated_normal"),
                                         seed = NULL) {
  if (get_keras_implementation() == "tensorflow" && tensorflow::tf_version() >= "2.0") {

    distribution <- match.arg(distribution)

    if (distribution == "normal")
      distribution <- "untruncated_normal"

    keras$initializers$VarianceScaling(
      scale = scale,
      mode = match.arg(mode),
      distribution = distribution,
      seed = as_nullable_integer(seed)
    )

  } else {
    keras$initializers$VarianceScaling(
      scale = scale,
      mode = match.arg(mode),
      distribution = match.arg(distribution),
      seed = as_nullable_integer(seed)
    )
  }
}


#' Initializer that generates a random orthogonal matrix.
#'
#' @inheritParams initializer_random_normal
#'
#' @param gain Multiplicative factor to apply to the orthogonal matrix.
#'
#' @section References:
#' Saxe et al., <https://arxiv.org/abs/1312.6120>
#'
#' @family initializers
#'
#' @export
initializer_orthogonal <- function(gain = 1.0, seed = NULL) {
  keras$initializers$Orthogonal(
    gain = gain,
    seed = as_nullable_integer(seed)
  )
}


#' Initializer that generates the identity matrix.
#'
#' Only use for square 2D matrices.
#'
#' @param gain Multiplicative factor to apply to the identity matrix
#'
#' @family initializers
#'
#' @export
initializer_identity <- function(gain = 1.0) {
  keras$initializers$Identity(
    gain = gain
  )
}

#' LeCun normal initializer.
#'
#' It draws samples from a truncated normal distribution centered on 0 with
#' `stddev <- sqrt(1 / fan_in)` where `fan_in` is the number of input units in
#' the weight tensor..
#'
#' @param seed A Python integer. Used to seed the random generator.
#'
#' @section References:
#'  - [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)
#'  - Efficient Backprop, \cite{LeCun, Yann et al. 1998}
#'
#' @family initializers
#'
#' @export
initializer_lecun_normal <- function(seed = NULL) {
  keras$initializers$lecun_normal(
    seed = as_nullable_integer(seed)
  )
}



#' Glorot normal initializer, also called Xavier normal initializer.
#'
#' It draws samples from a truncated normal distribution centered on 0
#' with `stddev = sqrt(2 / (fan_in + fan_out))`
#' where `fan_in` is the number of input units in the weight tensor
#' and `fan_out` is the number of output units in the weight tensor.
#'
#' @inheritParams initializer_random_normal
#'
#' @section References:
#' Glorot & Bengio, AISTATS 2010 <https://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf>
#'
#' @family initializers
#'
#' @export
initializer_glorot_normal <- function(seed = NULL) {
  keras$initializers$glorot_normal(
    seed = as_nullable_integer(seed)
  )
}


#' Glorot uniform initializer, also called Xavier uniform initializer.
#'
#' It draws samples from a uniform distribution within `-limit, limit`
#' where `limit` is `sqrt(6 / (fan_in + fan_out))`
#' where `fan_in` is the number of input units in the weight tensor
#' and `fan_out` is the number of output units in the weight tensor.
#'
#' @inheritParams initializer_random_normal
#'
#' @section References:
#' Glorot & Bengio, AISTATS 2010 https://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf
#'
#' @family initializers
#'
#' @export
initializer_glorot_uniform <- function(seed = NULL) {
  keras$initializers$glorot_uniform(
    seed = as_nullable_integer(seed)
  )
}


#' He normal initializer.
#'
#' It draws samples from a truncated normal distribution centered on 0 with
#' `stddev = sqrt(2 / fan_in)` where `fan_in` is the number of input units in
#' the weight tensor.
#'
#' @inheritParams initializer_random_normal
#'
#' @section References: He et al., https://arxiv.org/abs/1502.01852
#'
#' @family initializers
#'
#' @export
initializer_he_normal <- function(seed = NULL) {
  keras$initializers$he_normal(
    seed = seed
  )
}

#' He uniform variance scaling initializer.
#'
#' It draws samples from a uniform distribution within `-limit, limit` where
#' `limit`` is `sqrt(6 / fan_in)` where  `fan_in` is the number of input units in the
#' weight tensor.
#'
#' @inheritParams initializer_random_normal
#'
#' @section References: He et al., https://arxiv.org/abs/1502.01852
#'
#' @family initializers
#'
#' @export
initializer_he_uniform <- function(seed = NULL) {
  keras$initializers$he_uniform(
    seed = as_nullable_integer(seed)
  )
}

#' LeCun uniform initializer.
#'
#' It draws samples from a uniform distribution within `-limit, limit` where
#' `limit` is `sqrt(3 / fan_in)` where `fan_in` is the number of input units in
#' the weight tensor.
#'
#' @inheritParams initializer_random_normal
#'
#' @section References: LeCun 98, Efficient Backprop,
#'
#' @family initializers
#'
#' @export
initializer_lecun_uniform <- function(seed = NULL) {
  keras$initializers$lecun_uniform(
    seed = as_nullable_integer(seed)
  )
}
