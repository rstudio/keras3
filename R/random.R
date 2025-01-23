


#' Draws samples from a categorical distribution.
#'
#' @description
#' This function takes as input `logits`, a 2-D input tensor with shape
#' (batch_size, num_classes). Each row of the input represents a categorical
#' distribution, with each column index containing the log-probability for a
#' given class.
#'
#' The function will output a 2-D tensor with shape (batch_size, num_samples),
#' where each row contains samples from the corresponding row in `logits`.
#' Each column index contains an independent samples drawn from the input
#' distribution.
#'
#' @returns
#' A 2-D tensor with (batch_size, num_samples).
#'
#' @param logits
#' 2-D Tensor with shape (batch_size, num_classes). Each row
#' should define a categorical distribution with the unnormalized
#' log-probabilities for all classes.
#'
#' @param num_samples
#' Int, the number of independent samples to draw for each
#' row of the input. This will be the second dimension of the output
#' tensor's shape.
#'
#' @param dtype
#' Optional dtype of the output tensor.
#'
#' @param seed
#' Optional R integer or instance of
#' [`random_seed_generator()`].
#'  By default, the `seed` argument is `NULL`, and an internal global
#' [`random_seed_generator()`] is used. The `seed` argument can be
#' used to ensure deterministic (repeatable) random number generation.
#' Note that passing an integer as the `seed` value will produce the
#' same random values for each call. To generate different random
#' values for repeated calls, an instance of
#' [`random_seed_generator()`] must be provided as the `seed` value.
#'
#' Remark concerning the JAX backend: When tracing functions with the
#' JAX backend the global [`random_seed_generator()`] is not
#' supported. Therefore, during tracing the default value `seed=NULL`
#' will produce an error, and a `seed` argument must be provided.
#'
#' @export
#' @family random
# @seealso
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/random/categorical>
#' @tether keras.random.categorical
random_categorical <-
function (logits, num_samples, dtype = "int32", seed = NULL)
{
    args <- capture_args(list(num_samples = as_integer, seed = as_integer))
    do.call(keras$random$categorical, args)
}


#' Randomly set some values in a tensor to 0.
#'
#' @description
#' Randomly set some portion of values in the tensor to 0.
#'
#' @param seed
#' Initial seed for the random number generator
#'
#' @param inputs
#' A tensor
#'
#' @param rate
#' numeric
#'
#' @param noise_shape
#' A `shape()` value
#'
#' @returns A tensor that is a copy of `inputs` with some values set to `0`.
#' @export
#' @family random
# @seealso
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/random/dropout>
#' @tether keras.random.dropout
random_dropout <-
function (inputs, rate, noise_shape = NULL, seed = NULL)
{
    args <- capture_args(list(seed = as_integer, noise_shape = normalize_shape))
    do.call(keras$random$dropout, args)
}


#' Draw random samples from the Gamma distribution.
#'
#' @param shape
#' The shape of the random values to generate.
#'
#' @param alpha
#' Float, the parameter of the distribution.
#'
#' @param dtype
#' Optional dtype of the tensor. Only floating point types are
#' supported. If not specified, [`config_floatx()`] is used,
#' which defaults to `float32` unless you configured it otherwise (via
#' `config_set_floatx(float_dtype)`).
#'
#' @param seed
#' Optional R integer or instance of
#' [`random_seed_generator()`].
#'  By default, the `seed` argument is `NULL`, and an internal global
#' [`random_seed_generator()`] is used. The `seed` argument can be
#' used to ensure deterministic (repeatable) random number generation.
#' Note that passing an integer as the `seed` value will produce the
#' same random values for each call. To generate different random
#' values for repeated calls, an instance of
#' [`random_seed_generator()`] must be provided as the `seed` value.
#'
#' Remark concerning the JAX backend: When tracing functions with the
#' JAX backend the global [`random_seed_generator()`] is not
#' supported. Therefore, during tracing the default value `seed=NULL`
#' will produce an error, and a `seed` argument must be provided.
#'
#' @returns A tensor of random values.
#' @export
#' @family random
# @seealso
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/random/gamma>
#' @tether keras.random.gamma
random_gamma <-
function (shape, alpha, dtype = NULL, seed = NULL)
{
    args <- capture_args(list(shape = normalize_shape, seed = as_integer))
    do.call(keras$random$gamma, args)
}


#' Draw random integers from a uniform distribution.
#'
#' @description
#' The generated values follow a uniform distribution in the range
#' `[minval, maxval)`. The lower bound `minval` is included in the range,
#' while the upper bound `maxval` is excluded.
#'
#' `dtype` must be an integer type.
#'
#' @param shape
#' The shape of the random values to generate.
#'
#' @param minval
#' integer, lower bound of the range of
#' random values to generate (inclusive).
#'
#' @param maxval
#' integer, upper bound of the range of
#' random values to generate (exclusive).
#'
#' @param dtype
#' Optional dtype of the tensor. Only integer types are
#' supported. If not specified, `"int32"` is used.
#'
#' @param seed
#' Optional R integer or instance of
#' [`random_seed_generator()`].
#'  By default, the `seed` argument is `NULL`, and an internal global
#' [`random_seed_generator()`] is used. The `seed` argument can be
#' used to ensure deterministic (repeatable) random number generation.
#' Note that passing an integer as the `seed` value will produce the
#' same random values for each call. To generate different random
#' values for repeated calls, an instance of
#' [`random_seed_generator()`] must be provided as the `seed` value.
#'
#' Remark concerning the JAX backend: When tracing functions with the
#' JAX backend the global [`random_seed_generator()`] is not
#' supported. Therefore, during tracing the default value `seed=NULL`
#' will produce an error, and a `seed` argument must be provided.
#'
#' @returns A tensor of random values.
#' @export
#' @family random
# @seealso
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/random/randint>
#' @tether keras.random.randint
random_integer <-
function (shape, minval, maxval, dtype = "int32", seed = NULL)
{
    args <- capture_args(list(shape = normalize_shape, seed = as_integer,
        maxval = function (x)
        as_integer(ceiling(x)), minval = as_integer))
    do.call(keras$random$randint, args)
}


#' Draw random samples from a normal (Gaussian) distribution.
#'
#' @param shape
#' The shape of the random values to generate.
#'
#' @param mean
#' Float, defaults to 0. Mean of the random values to generate.
#'
#' @param stddev
#' Float, defaults to 1. Standard deviation of the random values
#' to generate.
#'
#' @param dtype
#' Optional dtype of the tensor. Only floating point types are
#' supported. If not specified, [`config_floatx()`] is used,
#' which defaults to `float32` unless you configured it otherwise (via
#' `config_set_floatx(float_dtype)`).
#'
#' @param seed
#' Optional R integer or instance of
#' [`random_seed_generator()`].
#'  By default, the `seed` argument is `NULL`, and an internal global
#' [`random_seed_generator()`] is used. The `seed` argument can be
#' used to ensure deterministic (repeatable) random number generation.
#' Note that passing an integer as the `seed` value will produce the
#' same random values for each call. To generate different random
#' values for repeated calls, an instance of
#' [`random_seed_generator()`] must be provided as the `seed` value.
#'
#' Remark concerning the JAX backend: When tracing functions with the
#' JAX backend the global [`random_seed_generator()`] is not
#' supported. Therefore, during tracing the default value `seed=NULL`
#' will produce an error, and a `seed` argument must be provided.
#'
#' @returns A tensor of random values.
#' @export
#' @family random
# @seealso
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/random/normal>
#' @tether keras.random.normal
random_normal <-
function (shape, mean = 0, stddev = 1, dtype = NULL, seed = NULL)
{
    args <- capture_args(list(shape = normalize_shape, seed = as_integer))
    do.call(keras$random$normal, args)
}


#' Shuffle the elements of a tensor uniformly at random along an axis.
#'
#' @param x
#' The tensor to be shuffled.
#'
#' @param axis
#' An integer specifying the axis along which to shuffle. Defaults to
#' `0`.
#'
#' @param seed
#' Optional R integer or instance of
#' [`random_seed_generator()`].
#'  By default, the `seed` argument is `NULL`, and an internal global
#' [`random_seed_generator()`] is used. The `seed` argument can be
#' used to ensure deterministic (repeatable) random number generation.
#' Note that passing an integer as the `seed` value will produce the
#' same random values for each call. To generate different random
#' values for repeated calls, an instance of
#' [`random_seed_generator()`] must be provided as the `seed` value.
#'
#' Remark concerning the JAX backend: When tracing functions with the
#' JAX backend the global [`random_seed_generator()`] is not
#' supported. Therefore, during tracing the default value `seed=NULL`
#' will produce an error, and a `seed` argument must be provided.
#'
#' @returns A tensor, a copy of `x` with the `axis` axis shuffled.
#' @export
#' @family random
# @seealso
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/random/shuffle>
#' @tether keras.random.shuffle
random_shuffle <-
function (x, axis = 1L, seed = NULL)
{
    args <- capture_args(list(axis = as_axis, seed = as_integer))
    do.call(keras$random$shuffle, args)
}


#' Draw samples from a truncated normal distribution.
#'
#' @description
#' The values are drawn from a normal distribution with specified mean and
#' standard deviation, discarding and re-drawing any samples that are more
#' than two standard deviations from the mean.
#'
#' @param shape
#' The shape of the random values to generate.
#'
#' @param mean
#' Float, defaults to 0. Mean of the random values to generate.
#'
#' @param stddev
#' Float, defaults to 1. Standard deviation of the random values
#' to generate.
#'
#' @param dtype
#' Optional dtype of the tensor. Only floating point types are
#' supported. If not specified, [`config_floatx()`] is used,
#' which defaults to `float32` unless you configured it otherwise (via
#' `config_set_floatx(float_dtype)`)
#'
#' @param seed
#' Optional R integer or instance of
#' [`random_seed_generator()`].
#'  By default, the `seed` argument is `NULL`, and an internal global
#' [`random_seed_generator()`] is used. The `seed` argument can be
#' used to ensure deterministic (repeatable) random number generation.
#' Note that passing an integer as the `seed` value will produce the
#' same random values for each call. To generate different random
#' values for repeated calls, an instance of
#' [`random_seed_generator()`] must be provided as the `seed` value.
#'
#' Remark concerning the JAX backend: When tracing functions with the
#' JAX backend the global [`random_seed_generator()`] is not
#' supported. Therefore, during tracing the default value `seed=NULL`
#' will produce an error, and a `seed` argument must be provided.
#'
#' @returns A tensor of random values.
#' @export
#' @family random
# @seealso
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/random/truncated_normal>
#' @tether keras.random.truncated_normal
random_truncated_normal <-
function (shape, mean = 0, stddev = 1, dtype = NULL, seed = NULL)
{
    args <- capture_args(list(shape = normalize_shape, seed = as_integer))
    do.call(keras$random$truncated_normal, args)
}


#' Draw samples from a uniform distribution.
#'
#' @description
#' The generated values follow a uniform distribution in the range
#' `[minval, maxval)`. The lower bound `minval` is included in the range,
#' while the upper bound `maxval` is excluded.
#'
#' `dtype` must be a floating point type, the default range is `[0, 1)`.
#'
#' @param shape
#' The shape of the random values to generate.
#'
#' @param minval
#' Float, defaults to 0. Lower bound of the range of
#' random values to generate (inclusive).
#'
#' @param maxval
#' Float, defaults to 1. Upper bound of the range of
#' random values to generate (exclusive).
#'
#' @param dtype
#' Optional dtype of the tensor. Only floating point types are
#' supported. If not specified, [`config_floatx()`] is used,
#' which defaults to `float32` unless you configured it otherwise (via
#' `config_set_floatx(float_dtype)`)
#'
#' @param seed
#' Optional R integer or instance of
#' [`random_seed_generator()`].
#'  By default, the `seed` argument is `NULL`, and an internal global
#' [`random_seed_generator()`] is used. The `seed` argument can be
#' used to ensure deterministic (repeatable) random number generation.
#' Note that passing an integer as the `seed` value will produce the
#' same random values for each call. To generate different random
#' values for repeated calls, an instance of
#' [`random_seed_generator()`] must be provided as the `seed` value.
#'
#' Remark concerning the JAX backend: When tracing functions with the
#' JAX backend the global [`random_seed_generator()`] is not
#' supported. Therefore, during tracing the default value `seed=NULL`
#' will produce an error, and a `seed` argument must be provided.
#'
#' @returns A tensor of random values.
#' @export
#' @family random
# @seealso
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/random/uniform>
#' @tether keras.random.uniform
random_uniform <-
function (shape, minval = 0, maxval = 1, dtype = NULL, seed = NULL)
{
    args <- capture_args(list(shape = normalize_shape, seed = as_integer))
    do.call(keras$random$uniform, args)
}


#' Generates variable seeds upon each call to a function generating random numbers.
#'
#' @description
#'
#' In Keras, all random number generators (such as
#' [`random_normal()`]) are stateless, meaning that if you pass an
#' integer seed to them (such as `seed=42`), they will return the same
#' values for repeated calls. To get different values for each
#' call, a `SeedGenerator` providing the state of the random generator
#' has to be used.
#'
#' Note that all the random number generators have a default seed of `NULL`,
#' which implies that an internal global `SeedGenerator` is used.
#' If you need to decouple the RNG from the global state you can provide
#' a local `StateGenerator` with either a deterministic or random initial
#' state.
#'
#' Remark concerning the JAX backen: Note that the use of a local
#' `StateGenerator` as seed argument is required for JIT compilation of
#' RNG with the JAX backend, because the use of global state is not
#' supported.
#'
#' # Examples
#' ```{r}
#' seed_gen <- random_seed_generator(seed = 42)
#' values <- random_normal(shape = c(2, 3), seed = seed_gen)
#' new_values <- random_normal(shape = c(2, 3), seed = seed_gen)
#' ```
#'
#' Usage in a layer:
#'
#' ```{r}
#' layer_dropout2 <- new_layer_class(
#'   "dropout2",
#'   initialize = function(...) {
#'     super$initialize(...)
#'     self$seed_generator <- random_seed_generator(seed = 1337)
#'   },
#'   call = function(x, training = FALSE) {
#'     if (training) {
#'       return(random_dropout(x, rate = 0.5, seed = self$seed_generator))
#'     }
#'     return(x)
#'   }
#' )
#'
#' out <- layer_dropout(rate = 0.8)
#' out(op_ones(10), training = TRUE)
#' ```
#'
#' @param seed
#' Initial seed for the random number generator
#'
#' @param name String, name for the object
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @returns A `SeedGenerator` instance, which can be passed as the `seed = `
#'   argument to other random tensor generators.
#' @export
#' @family random
# @seealso
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/random/SeedGenerator>
#'
#' @tether keras.random.SeedGenerator
random_seed_generator <-
function (seed = NULL, name = NULL, ...)
{
    args <- capture_args(list(seed = as_integer))
    do.call(keras$random$SeedGenerator, args)
}

#' Draw samples from a Beta distribution.
#'
#' @description
#' The values are drawn from a Beta distribution parametrized
#' by alpha and beta.
#'
#' @param shape
#' The shape of the random values to generate.
#'
#' @param alpha
#' Float or an array of floats representing the first
#' parameter alpha. Must be broadcastable with `beta` and `shape`.
#'
#' @param beta
#' Float or an array of floats representing the second
#' parameter beta. Must be broadcastable with `alpha` and `shape`.
#'
#' @param dtype
#' Optional dtype of the tensor. Only floating point types are
#' supported. If not specified, `config_floatx()` is used,
#' which defaults to `"float32"` unless you configured it otherwise (via
#' `config_set_floatx(float_dtype)`).
#'
#' @param seed
#' Optional R integer or instance of
#' [`random_seed_generator()`].
#'  By default, the `seed` argument is `NULL`, and an internal global
#' [`random_seed_generator()`] is used. The `seed` argument can be
#' used to ensure deterministic (repeatable) random number generation.
#' Note that passing an integer as the `seed` value will produce the
#' same random values for each call. To generate different random
#' values for repeated calls, an instance of
#' [`random_seed_generator()`] must be provided as the `seed` value.
#'
#' Remark concerning the JAX backend: When tracing functions with the
#' JAX backend the global [`random_seed_generator()`] is not
#' supported. Therefore, during tracing the default value `seed=NULL`
#' will produce an error, and a `seed` argument must be provided.
#'
#' @export
#' @returns A tensor of random values.
#' @family random
#' @tether keras.random.beta
random_beta <-
function (shape, alpha, beta, dtype = NULL, seed = NULL)
{
  args <- capture_args(list(shape = normalize_shape, seed = as_integer))
  do.call(keras$random$beta, args)
}


#' Draw samples from a Binomial distribution.
#'
#' @description
#' The values are drawn from a Binomial distribution with
#' specified trial count and probability of success.
#'
#' @param shape
#' The shape of the random values to generate.
#'
#' @param counts
#' A number or array of numbers representing the
#' number of trials. It must be broadcastable with `probabilities`.
#'
#' @param probabilities
#' A float or array of floats representing the
#' probability of success of an individual event.
#' It must be broadcastable with `counts`.
#'
#' @param dtype
#' Optional dtype of the tensor. Only floating point types are
#' supported. If not specified, `config_floatx()` is used,
#' which defaults to `"float32"` unless you configured it otherwise (via
#' `config_set_floatx(float_dtype)`).
#'
#' @param seed
#' Optional R integer or instance of
#' [`random_seed_generator()`].
#'  By default, the `seed` argument is `NULL`, and an internal global
#' [`random_seed_generator()`] is used. The `seed` argument can be
#' used to ensure deterministic (repeatable) random number generation.
#' Note that passing an integer as the `seed` value will produce the
#' same random values for each call. To generate different random
#' values for repeated calls, an instance of
#' [`random_seed_generator()`] must be provided as the `seed` value.
#'
#' Remark concerning the JAX backend: When tracing functions with the
#' JAX backend the global [`random_seed_generator()`] is not
#' supported. Therefore, during tracing the default value `seed=NULL`
#' will produce an error, and a `seed` argument must be provided.
#'
#' @export
#' @returns A tensor of random values.
#' @family random
#' @tether keras.random.binomial
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/random/binomial>
random_binomial <-
function (shape, counts, probabilities, dtype = NULL, seed = NULL)
{
    args <- capture_args(list(shape = normalize_shape, seed = as_integer))
    do.call(keras$random$binomial, args)
}
