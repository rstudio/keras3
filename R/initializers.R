
#' Initializer that generates tensors with constant values.
#'
#' @description
#' Only scalar values are allowed.
#' The constant value provided must be convertible to the dtype requested
#' when calling the initializer.
#'
#' # Examples
#' ```{r}
#' # Standalone usage:
#' initializer <- initializer_constant(10)
#' values <- initializer(shape = c(2, 2))
#' ```
#'
#' ```{r}
#' # Usage in a Keras layer:
#' initializer <- initializer_constant(10)
#' layer <- layer_dense(units = 3, kernel_initializer = initializer)
#' ```
#'
#' @param value
#' A numeric scalar.
#'
#' @returns An `Initializer` instance that can be passed to layer or variable
#'   constructors, or called directly with a `shape` to return a Tensor.
#' @export
#' @family constant initializers
#' @family initializers
#' @seealso
#' + <https://keras.io/api/layers/initializers#constant-class>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/initializers/Constant>
#' @tether keras.initializers.Constant
initializer_constant <-
function (value = 0)
{
    args <- capture_args()
    do.call(keras$initializers$Constant, args)
}


#' Initializer that generates the identity matrix.
#'
#' @description
#' Only usable for generating 2D matrices.
#'
#' # Examples
#' ```{r}
#' # Standalone usage:
#' initializer <- initializer_identity()
#' values <- initializer(shape = c(2, 2))
#' ```
#'
#' ```{r}
#' # Usage in a Keras layer:
#' initializer <- initializer_identity()
#' layer <- layer_dense(units = 3, kernel_initializer = initializer)
#' ```
#'
#' @param gain
#' Multiplicative factor to apply to the identity matrix.
#'
#' @inherit initializer_constant return
#' @export
#' @family constant initializers
#' @family initializers
# @seealso
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/initializers/IdentityInitializer>
#' @tether keras.initializers.IdentityInitializer
initializer_identity <-
function (gain = 1)
{
    args <- capture_args()
    do.call(keras$initializers$IdentityInitializer, args)
}


#' Initializer that generates tensors initialized to 1.
#'
#' @description
#' Also available via the shortcut function `ones`.
#'
#' # Examples
#' ```{r}
#' # Standalone usage:
#' initializer <- initializer_ones()
#' values <- initializer(shape = c(2, 2))
#' ```
#'
#' ```{r}
#' # Usage in a Keras layer:
#' initializer <- initializer_ones()
#' layer <- layer_dense(units = 3, kernel_initializer = initializer)
#' ```
#'
#' @inherit initializer_constant return
#' @export
#' @family constant initializers
#' @family initializers
#' @seealso
#' + <https://keras.io/api/layers/initializers#ones-class>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/initializers/Ones>
#' @tether keras.initializers.Ones
initializer_ones <-
function ()
{
    args <- capture_args()
    do.call(keras$initializers$Ones, args)
}


#' Initializer that generates tensors initialized to 0.
#'
#' @description
#'
#' # Examples
#' ```{r}
#' # Standalone usage:
#' initializer <- initializer_zeros()
#' values <- initializer(shape = c(2, 2))
#' ```
#'
#' ```{r}
#' # Usage in a Keras layer:
#' initializer <- initializer_zeros()
#' layer <- layer_dense(units = 3, kernel_initializer = initializer)
#' ```
#'
#' @inherit initializer_constant return
#' @export
#' @family constant initializers
#' @family initializers
#' @seealso
#' + <https://keras.io/api/layers/initializers#zeros-class>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/initializers/Zeros>
#' @tether keras.initializers.Zeros
initializer_zeros <-
function ()
{
    args <- capture_args()
    do.call(keras$initializers$Zeros, args)
}


#' The Glorot normal initializer, also called Xavier normal initializer.
#'
#' @description
#' Draws samples from a truncated normal distribution centered on 0 with
#' `stddev = sqrt(2 / (fan_in + fan_out))` where `fan_in` is the number of
#' input units in the weight tensor and `fan_out` is the number of output units
#' in the weight tensor.
#'
#' # Examples
#' ```{r}
#' # Standalone usage:
#' initializer <- initializer_glorot_normal()
#' values <- initializer(shape = c(2, 2))
#' ```
#'
#' ```{r}
#' # Usage in a Keras layer:
#' initializer <- initializer_glorot_normal()
#' layer <- layer_dense(units = 3, kernel_initializer = initializer)
#' ```
#'
#' # Reference
#' - [Glorot et al., 2010](https://proceedings.mlr.press/v9/glorot10a.html)
#'
#' @param seed
#' An integer or instance of
#' `random_seed_generator()`.
#' Used to make the behavior of the initializer
#' deterministic. Note that an initializer seeded with an integer
#' or `NULL` (unseeded) will produce the same random values
#' across multiple calls. To get different random values
#' across multiple calls, use as seed an instance
#' of `random_seed_generator()`.
#'
#' @inherit initializer_constant return
#' @export
#' @family random initializers
#' @family initializers
#' @seealso
#' + <https://keras.io/api/layers/initializers#glorotnormal-class>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/initializers/GlorotNormal>
#' @tether keras.initializers.GlorotNormal
initializer_glorot_normal <-
function (seed = NULL)
{
    args <- capture_args(list(seed = as_integer))
    do.call(keras$initializers$GlorotNormal, args)
}


#' The Glorot uniform initializer, also called Xavier uniform initializer.
#'
#' @description
#' Draws samples from a uniform distribution within `[-limit, limit]`, where
#' `limit = sqrt(6 / (fan_in + fan_out))` (`fan_in` is the number of input
#' units in the weight tensor and `fan_out` is the number of output units).
#'
#' # Examples
#' ```{r}
#' # Standalone usage:
#' initializer <- initializer_glorot_uniform()
#' values <- initializer(shape = c(2, 2))
#' ```
#'
#' ```{r}
#' # Usage in a Keras layer:
#' initializer <- initializer_glorot_uniform()
#' layer <- layer_dense(units = 3, kernel_initializer = initializer)
#' ```
#'
#' # Reference
#' - [Glorot et al., 2010](https://proceedings.mlr.press/v9/glorot10a.html)
#'
#' @param seed
#' An integer or instance of
#' `random_seed_generator()`.
#' Used to make the behavior of the initializer
#' deterministic. Note that an initializer seeded with an integer
#' or `NULL` (unseeded) will produce the same random values
#' across multiple calls. To get different random values
#' across multiple calls, use as seed an instance
#' of `random_seed_generator()`.
#'
#' @inherit initializer_constant return
#' @export
#' @family random initializers
#' @family initializers
#' @seealso
#' + <https://keras.io/api/layers/initializers#glorotuniform-class>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/initializers/GlorotUniform>
#' @tether keras.initializers.GlorotUniform
initializer_glorot_uniform <-
function (seed = NULL)
{
    args <- capture_args(list(seed = as_integer))
    do.call(keras$initializers$GlorotUniform, args)
}


#' He normal initializer.
#'
#' @description
#' It draws samples from a truncated normal distribution centered on 0 with
#' `stddev = sqrt(2 / fan_in)` where `fan_in` is the number of input units in
#' the weight tensor.
#'
#' # Examples
#' ```{r}
#' # Standalone usage:
#' initializer <- initializer_he_normal()
#' values <- initializer(shape = c(2, 2))
#' ```
#'
#' ```{r}
#' # Usage in a Keras layer:
#' initializer <- initializer_he_normal()
#' layer <- layer_dense(units = 3, kernel_initializer = initializer)
#' ```
#'
#' # Reference
#' - [He et al., 2015](https://arxiv.org/abs/1502.01852)
#'
#' @param seed
#' An integer or instance of
#' `random_seed_generator()`.
#' Used to make the behavior of the initializer
#' deterministic. Note that an initializer seeded with an integer
#' or `NULL` (unseeded) will produce the same random values
#' across multiple calls. To get different random values
#' across multiple calls, use as seed an instance
#' of `random_seed_generator()`.
#'
#' @inherit initializer_constant return
#' @export
#' @family random initializers
#' @family initializers
#' @seealso
#' + <https://keras.io/api/layers/initializers#henormal-class>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/initializers/HeNormal>
#' @tether keras.initializers.HeNormal
initializer_he_normal <-
function (seed = NULL)
{
    args <- capture_args(list(seed = as_integer))
    do.call(keras$initializers$HeNormal, args)
}


#' He uniform variance scaling initializer.
#'
#' @description
#' Draws samples from a uniform distribution within `[-limit, limit]`, where
#' `limit = sqrt(6 / fan_in)` (`fan_in` is the number of input units in the
#' weight tensor).
#'
#' # Examples
#' ```{r}
#' # Standalone usage:
#' initializer <- initializer_he_uniform()
#' values <- initializer(shape = c(2, 2))
#' ```
#'
#' ```{r}
#' # Usage in a Keras layer:
#' initializer <- initializer_he_uniform()
#' layer <- layer_dense(units = 3, kernel_initializer = initializer)
#' ```
#'
#' # Reference
#' - [He et al., 2015](https://arxiv.org/abs/1502.01852)
#'
#' @param seed
#' A integer or instance of
#' `random_seed_generator()`.
#' Used to make the behavior of the initializer
#' deterministic. Note that an initializer seeded with an integer
#' or `NULL` (unseeded) will produce the same random values
#' across multiple calls. To get different random values
#' across multiple calls, use as seed an instance
#' of `random_seed_generator()`.
#'
#' @inherit initializer_constant return
#' @export
#' @family random initializers
#' @family initializers
#' @seealso
#' + <https://keras.io/api/layers/initializers#heuniform-class>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/initializers/HeUniform>
#' @tether keras.initializers.HeUniform
initializer_he_uniform <-
function (seed = NULL)
{
    args <- capture_args(list(seed = as_integer))
    do.call(keras$initializers$HeUniform, args)
}


#' Lecun normal initializer.
#'
#' @description
#' Initializers allow you to pre-specify an initialization strategy, encoded in
#' the Initializer object, without knowing the shape and dtype of the variable
#' being initialized.
#'
#' Draws samples from a truncated normal distribution centered on 0 with
#' `stddev = sqrt(1 / fan_in)` where `fan_in` is the number of input units in
#' the weight tensor.
#'
#' # Examples
#' ```{r}
#' # Standalone usage:
#' initializer <- initializer_lecun_normal()
#' values <- initializer(shape = c(2, 2))
#' ```
#'
#' ```{r}
#' # Usage in a Keras layer:
#' initializer <- initializer_lecun_normal()
#' layer <- layer_dense(units = 3, kernel_initializer = initializer)
#' ```
#'
#' # Reference
#' - [Klambauer et al., 2017](https://arxiv.org/abs/1706.02515)
#'
#' @param seed
#' An integer or instance of
#' `random_seed_generator()`.
#' Used to make the behavior of the initializer
#' deterministic. Note that an initializer seeded with an integer
#' or `NULL` (unseeded) will produce the same random values
#' across multiple calls. To get different random values
#' across multiple calls, use as seed an instance
#' of `random_seed_generator()`.
#'
#' @inherit initializer_constant return
#' @export
#' @family random initializers
#' @family initializers
# @seealso
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/initializers/LecunNormal>
#' @tether keras.initializers.LecunNormal
initializer_lecun_normal <-
function (seed = NULL)
{
    args <- capture_args(list(seed = as_integer))
    do.call(keras$initializers$LecunNormal, args)
}


#' Lecun uniform initializer.
#'
#' @description
#' Draws samples from a uniform distribution within `[-limit, limit]`, where
#' `limit = sqrt(3 / fan_in)` (`fan_in` is the number of input units in the
#' weight tensor).
#'
#' # Examples
#' ```{r}
#' # Standalone usage:
#' initializer <- initializer_lecun_uniform()
#' values <- initializer(shape = c(2, 2))
#' ```
#'
#' ```{r}
#' # Usage in a Keras layer:
#' initializer <- initializer_lecun_uniform()
#' layer <- layer_dense(units = 3, kernel_initializer = initializer)
#' ```
#'
#' # Reference
#' - [Klambauer et al., 2017](https://arxiv.org/abs/1706.02515)
#'
#' @param seed
#' An integer or instance of
#' `random_seed_generator()`.
#' Used to make the behavior of the initializer
#' deterministic. Note that an initializer seeded with an integer
#' or `NULL` (unseeded) will produce the same random values
#' across multiple calls. To get different random values
#' across multiple calls, use as seed an instance
#' of `random_seed_generator()`.
#'
#' @inherit initializer_constant return
#' @export
#' @family random initializers
#' @family initializers
# @seealso
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/initializers/LecunUniform>
#' @tether keras.initializers.LecunUniform
initializer_lecun_uniform <-
function (seed = NULL)
{
    args <- capture_args(list(seed = as_integer))
    do.call(keras$initializers$LecunUniform, args)
}


#' Initializer that generates an orthogonal matrix.
#'
#' @description
#' If the shape of the tensor to initialize is two-dimensional, it is
#' initialized with an orthogonal matrix obtained from the QR decomposition of
#' a matrix of random numbers drawn from a normal distribution. If the matrix
#' has fewer rows than columns then the output will have orthogonal rows.
#' Otherwise, the output will have orthogonal columns.
#'
#' If the shape of the tensor to initialize is more than two-dimensional,
#' a matrix of shape `(shape[1] * ... * shape[n - 1], shape[n])`
#' is initialized, where `n` is the length of the shape vector.
#' The matrix is subsequently reshaped to give a tensor of the desired shape.
#'
#' # Examples
#' ```{r}
#' # Standalone usage:
#' initializer <- initializer_orthogonal()
#' values <- initializer(shape = c(2, 2))
#' ```
#'
#' ```{r}
#' # Usage in a Keras layer:
#' initializer <- initializer_orthogonal()
#' layer <- layer_dense(units = 3, kernel_initializer = initializer)
#' ```
#'
#' # Reference
#' - [Saxe et al., 2014](https://openreview.net/forum?id=_wzZwKpTDF_9C)
#'
#' @param gain
#' Multiplicative factor to apply to the orthogonal matrix.
#'
#' @param seed
#' An integer. Used to make the behavior of the initializer
#' deterministic.
#'
#' @inherit initializer_constant return
#' @export
#' @family random initializers
#' @family initializers
# @seealso
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/initializers/OrthogonalInitializer>
#' @tether keras.initializers.OrthogonalInitializer
initializer_orthogonal <-
function (gain = 1, seed = NULL)
{
    args <- capture_args(list(seed = as_integer))
    do.call(keras$initializers$OrthogonalInitializer, args)
}


#' Random normal initializer.
#'
#' @description
#' Draws samples from a normal distribution for given parameters.
#'
#' # Examples
#' ```{r}
#' # Standalone usage:
#' initializer <- initializer_random_normal(mean = 0.0, stddev = 1.0)
#' values <- initializer(shape = c(2, 2))
#' ```
#'
#' ```{r}
#' # Usage in a Keras layer:
#' initializer <- initializer_random_normal(mean = 0.0, stddev = 1.0)
#' layer <- layer_dense(units = 3, kernel_initializer = initializer)
#' ```
#'
#' @param mean
#' A numeric scalar. Mean of the random
#' values to generate.
#'
#' @param stddev
#' A numeric scalar. Standard deviation of
#' the random values to generate.
#'
#' @param seed
#' An integer or instance of
#' `random_seed_generator()`.
#' Used to make the behavior of the initializer
#' deterministic. Note that an initializer seeded with an integer
#' or `NULL` (unseeded) will produce the same random values
#' across multiple calls. To get different random values
#' across multiple calls, use as seed an instance
#' of `random_seed_generator()`.
#'
#' @inherit initializer_constant return
#' @export
#' @family random initializers
#' @family initializers
#' @seealso
#' + <https://keras.io/api/layers/initializers#randomnormal-class>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/initializers/RandomNormal>
#' @tether keras.initializers.RandomNormal
initializer_random_normal <-
function (mean = 0, stddev = 0.05, seed = NULL)
{
    args <- capture_args(list(seed = as_integer))
    do.call(keras$initializers$RandomNormal, args)
}


#' Random uniform initializer.
#'
#' @description
#' Draws samples from a uniform distribution for given parameters.
#'
#' # Examples
#' ```{r}
#' # Standalone usage:
#' initializer <- initializer_random_uniform(minval = 0.0, maxval = 1.0)
#' values <- initializer(shape = c(2, 2))
#' ```
#'
#' ```{r}
#' # Usage in a Keras layer:
#' initializer <- initializer_random_uniform(minval = 0.0, maxval = 1.0)
#' layer <- layer_dense(units = 3, kernel_initializer = initializer)
#' ```
#'
#' @param minval
#' A numeric scalar or a scalar keras tensor. Lower bound of the
#' range of random values to generate (inclusive).
#'
#' @param maxval
#' A numeric scalar or a scalar keras tensor. Upper bound of the
#' range of random values to generate (exclusive).
#'
#' @param seed
#' An integer or instance of
#' `random_seed_generator()`.
#' Used to make the behavior of the initializer
#' deterministic. Note that an initializer seeded with an integer
#' or `NULL` (unseeded) will produce the same random values
#' across multiple calls. To get different random values
#' across multiple calls, use as seed an instance
#' of `random_seed_generator()`.
#'
#' @inherit initializer_constant return
#' @export
#' @family random initializers
#' @family initializers
#' @seealso
#' + <https://keras.io/api/layers/initializers#randomuniform-class>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/initializers/RandomUniform>
#' @tether keras.initializers.RandomUniform
initializer_random_uniform <-
function (minval = -0.05, maxval = 0.05, seed = NULL)
{
    args <- capture_args(list(seed = as_integer))
    do.call(keras$initializers$RandomUniform, args)
}


#' Initializer that generates a truncated normal distribution.
#'
#' @description
#' The values generated are similar to values from a
#' `RandomNormal` initializer, except that values more
#' than two standard deviations from the mean are
#' discarded and re-drawn.
#'
#' # Examples
#' ```{r}
#' # Standalone usage:
#' initializer <- initializer_truncated_normal(mean = 0, stddev = 1)
#' values <- initializer(shape = c(2, 2))
#' ```
#'
#' ```{r}
#' # Usage in a Keras layer:
#' initializer <- initializer_truncated_normal(mean = 0, stddev = 1)
#' layer <- layer_dense(units = 3, kernel_initializer = initializer)
#' ```
#'
#' @param mean
#' A numeric scalar. Mean of the random
#' values to generate.
#'
#' @param stddev
#' A numeric scalar. Standard deviation of
#' the random values to generate.
#'
#' @param seed
#' An integer or instance of
#' `random_seed_generator()`.
#' Used to make the behavior of the initializer
#' deterministic. Note that an initializer seeded with an integer
#' or `NULL` (unseeded) will produce the same random values
#' across multiple calls. To get different random values
#' across multiple calls, use as seed an instance
#' of `random_seed_generator()`.
#'
#' @inherit initializer_constant return
#' @export
#' @family random initializers
#' @family initializers
#' @seealso
#' + <https://keras.io/api/layers/initializers#truncatednormal-class>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/initializers/TruncatedNormal>
#' @tether keras.initializers.TruncatedNormal
initializer_truncated_normal <-
function (mean = 0, stddev = 0.05, seed = NULL)
{
    args <- capture_args(list(seed = as_integer))
    do.call(keras$initializers$TruncatedNormal, args)
}


#' Initializer that adapts its scale to the shape of its input tensors.
#'
#' @description
#' With `distribution = "truncated_normal" or "untruncated_normal"`, samples are
#' drawn from a truncated/untruncated normal distribution with a mean of zero
#' and a standard deviation (after truncation, if used) `stddev = sqrt(scale /
#' n)`, where `n` is:
#'
#' - number of input units in the weight tensor, if `mode = "fan_in"`
#' - number of output units, if `mode = "fan_out"`
#' - average of the numbers of input and output units, if `mode = "fan_avg"`
#'
#' With `distribution = "uniform"`, samples are drawn from a uniform distribution
#' within `[-limit, limit]`, where `limit = sqrt(3 * scale / n)`.
#'
#' # Examples
#' ```{r}
#' # Standalone usage:
#' initializer <- initializer_variance_scaling(scale = 0.1, mode = 'fan_in',
#'                                             distribution = 'uniform')
#' values <- initializer(shape = c(2, 2))
#' ```
#'
#' ```{r}
#' # Usage in a Keras layer:
#' initializer <- initializer_variance_scaling(scale = 0.1, mode = 'fan_in',
#'                                             distribution = 'uniform')
#' layer <- layer_dense(units = 3, kernel_initializer = initializer)
#' ```
#'
#' @param scale
#' Scaling factor (positive float).
#'
#' @param mode
#' One of `"fan_in"`, `"fan_out"`, `"fan_avg"`.
#'
#' @param distribution
#' Random distribution to use.
#' One of `"truncated_normal"`, `"untruncated_normal"`, or `"uniform"`.
#'
#' @param seed
#' An integer or instance of
#' `random_seed_generator()`.
#' Used to make the behavior of the initializer
#' deterministic. Note that an initializer seeded with an integer
#' or `NULL` (unseeded) will produce the same random values
#' across multiple calls. To get different random values
#' across multiple calls, use as seed an instance
#' of `random_seed_generator()`.
#'
#' @inherit initializer_constant return
#' @export
#' @family random initializers
#' @family initializers
#' @seealso
#' + <https://keras.io/api/layers/initializers#variancescaling-class>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/initializers/VarianceScaling>
#' @tether keras.initializers.VarianceScaling
initializer_variance_scaling <-
function (scale = 1, mode = "fan_in", distribution = "truncated_normal",
    seed = NULL)
{
    args <- capture_args(list(seed = as_integer))
    do.call(keras$initializers$VarianceScaling, args)
}

#' Initializer of Conv kernels for Short-term Fourier Transformation (STFT).
#'
#' @description
#' Since the formula involves complex numbers, this class compute either the
#' real or the imaginary components of the final output.
#'
#' Additionally, this initializer supports windowing functions across the time
#' dimension as commonly used in STFT. Windowing functions from the Python module
#' `scipy.signal.windows` are supported, including the common `hann` and
#' `hamming` windowing functions. This layer supports periodic windows and
#' scaling-based normalization.
#'
#' This is primarily intended for use in the `STFTSpectrogram` layer.
#'
#' # Examples
#' ```{r}
#' # Standalone usage:
#' initializer <- initializer_stft("real", "hann", "density", FALSE)
#' values <- initializer(shape = c(128, 1, 513))
#' ```
#'
#' @param side
#' String, `"real"` or `"imag"` deciding if the kernel will compute
#' the real side or the imaginary side of the output. Defaults to
#' `"real"`.
#'
#' @param window
#' String for the name of the windowing function in the
#' `scipy.signal.windows` module, or array_like for the window values,
#' or `NULL` for no windowing.
#'
#' @param scaling
#' String, `"density"` or `"spectrum"` for scaling of the window
#' for normalization, either L2 or L1 normalization.
#' `NULL` for no scaling.
#'
#' @param periodic
#' Boolean, if True, the window function will be treated as
#' periodic. Defaults to `FALSE`.
#'
#' @export
#' @inherit initializer_constant return
#' @family initializers
#' @family constant initializers
#' @tether keras.initializers.STFT
initializer_stft <-
function (side = "real", window = "hann", scaling = "density",
    periodic = FALSE)
{
    args <- capture_args(NULL)
    do.call(keras$initializers$STFT, args)
}



#' @export
py_to_r_wrapper.keras.src.initializers.initializer.Initializer <- function(x) {
    force(x)
    as.function.default(c(formals(x), quote({
        args <- capture_args(list(shape = normalize_shape))
        do.call(x, args)
    })))
}
