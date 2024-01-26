


#' Exponential Linear Unit.
#'
#' @description
#' The exponential linear unit (ELU) with `alpha > 0` is defined as:
#'
#' - `x` if `x > 0`
#' - `alpha * exp(x) - 1` if `x < 0`
#'
#' ELUs have negative values which pushes the mean of the activations
#' closer to zero.
#'
#' Mean activations that are closer to zero enable faster learning as they
#' bring the gradient closer to the natural gradient.
#' ELUs saturate to a negative value when the argument gets smaller.
#' Saturation means a small derivative which decreases the variation
#' and the information that is propagated to the next layer.
#'
#' # Reference
#' - [Clevert et al., 2016](https://arxiv.org/abs/1511.07289)
#'
#' @param x
#' Input tensor.
#'
#' @param alpha
#' Numeric. See description for details.
#'
#' @export
#' @family activations
#' @seealso
#' + <https://keras.io/api/layers/activations#elu-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/activations/elu>
#' @tether keras.activations.elu
activation_elu <-
function (x, alpha = 1)
{
    args <- capture_args()
    do.call(keras$activations$elu, args)
}


#' Exponential activation function.
#'
#' @param x
#' Input tensor.
#'
#' @export
#' @family activations
#' @seealso
#' + <https://keras.io/api/layers/activations#exponential-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/activations/exponential>
#' @tether keras.activations.exponential
activation_exponential <-
function (x)
{
    args <- capture_args()
    do.call(keras$activations$exponential, args)
}


#' Gaussian error linear unit (GELU) activation function.
#'
#' @description
#' The Gaussian error linear unit (GELU) is defined as:
#'
#' `gelu(x) = x * P(X <= x)` where `P(X) ~ N(0, 1)`,
#' i.e. `gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2)))`.
#'
#' GELU weights inputs by their value, rather than gating
#' inputs by their sign as in ReLU.
#'
#' # Reference
#' - [Hendrycks et al., 2016](https://arxiv.org/abs/1606.08415)
#'
#' @param x
#' Input tensor.
#'
#' @param approximate
#' A `bool`, whether to enable approximation.
#'
#' @export
#' @family activations
#' @seealso
#' + <https://keras.io/api/layers/activations#gelu-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/activations/gelu>
#' @tether keras.activations.gelu
activation_gelu <-
function (x, approximate = FALSE)
{
    args <- capture_args()
    do.call(keras$activations$gelu, args)
}


#' Hard sigmoid activation function.
#'
#' @description
#' The hard sigmoid activation is defined as:
#'
#' - `0` if `if x < -2.5`
#' - `1` if `x > 2.5`
#' - `0.2 * x + 0.5` if `-2.5 <= x <= 2.5`
#'
#' It's a faster, piecewise linear approximation
#' of the sigmoid activation.
#'
#' # Reference
#' - [Wikipedia "Hard sigmoid"](https://en.wikipedia.org/wiki/Hard_sigmoid)
#'
#' @param x
#' Input tensor.
#'
#' @export
#' @family activations
#' @seealso
#' + <https://keras.io/api/layers/activations#hardsigmoid-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/activations/hard_sigmoid>
#' @tether keras.activations.hard_sigmoid
activation_hard_sigmoid <-
function (x)
{
    args <- capture_args()
    do.call(keras$activations$hard_sigmoid, args)
}


#' Leaky relu activation function.
#'
#' @param x
#' Input tensor.
#'
#' @param negative_slope
#' A `float` that controls the slope
#' for values lower than the threshold.
#'
#' @export
#' @family activations
#' @seealso
#' + <https://keras.io/api/layers/activations#leakyrelu-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/activations/leaky_relu>
#' @tether keras.activations.leaky_relu
activation_leaky_relu <-
function (x, negative_slope = 0.2)
{
    args <- capture_args()
    do.call(keras$activations$leaky_relu, args)
}


#' Linear activation function (pass-through).
#'
#' @description
#' A "linear" activation is an identity function:
#' it returns the input, unmodified.
#'
#' @param x
#' Input tensor.
#'
#' @export
#' @family activations
#' @seealso
#' + <https://keras.io/api/layers/activations#linear-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/activations/linear>
#' @tether keras.activations.linear
activation_linear <-
function (x)
{
    args <- capture_args()
    do.call(keras$activations$linear, args)
}


#' Log-Softmax activation function.
#'
#' @description
#' Each input vector is handled independently.
#' The `axis` argument sets which axis of the input the function
#' is applied along.
#'
#' @param x
#' Input tensor.
#'
#' @param axis
#' Integer, axis along which the softmax is applied.
#'
#' @export
#' @family activations
#' @seealso
#' + <https://keras.io/api/layers/activations#logsoftmax-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/activations/log_softmax>
#' @tether keras.activations.log_softmax
activation_log_softmax <-
function (x, axis = -1L)
{
    args <- capture_args(list(axis = as_axis))
    do.call(keras$activations$log_softmax, args)
}


#' Mish activation function.
#'
#' @description
#' It is defined as:
#'
#' `mish(x) = x * tanh(softplus(x))`
#'
#' where `softplus` is defined as:
#'
#' `softplus(x) = log(exp(x) + 1)`
#'
#' # Reference
#' - [Misra, 2019](https://arxiv.org/abs/1908.08681)
#'
#' @param x
#' Input tensor.
#'
#' @export
#' @family activations
#' @seealso
#' + <https://keras.io/api/layers/activations#mish-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/activations/mish>
#' @tether keras.activations.mish
activation_mish <-
function (x)
{
    args <- capture_args()
    do.call(keras$activations$mish, args)
}


#' Applies the rectified linear unit activation function.
#'
#' @description
#' With default values, this returns the standard ReLU activation:
#' `max(x, 0)`, the element-wise maximum of 0 and the input tensor.
#'
#' Modifying default parameters allows you to use non-zero thresholds,
#' change the max value of the activation,
#' and to use a non-zero multiple of the input for values below the threshold.
#'
#' # Examples
#' ```{r}
#' x <- c(-10, -5, 0, 5, 10)
#' activation_relu(x)
#' activation_relu(x, negative_slope = 0.5)
#' activation_relu(x, max_value = 5)
#' activation_relu(x, threshold = 5)
#' ```
#'
#' @returns
#' A tensor with the same shape and dtype as input `x`.
#'
#' @param x
#' Input tensor.
#'
#' @param negative_slope
#' A `numeric` that controls the slope
#' for values lower than the threshold.
#'
#' @param max_value
#' A `numeric` that sets the saturation threshold (the largest
#' value the function will return).
#'
#' @param threshold
#' A `numeric` giving the threshold value of the activation
#' function below which values will be damped or set to zero.
#'
#' @export
#' @family activations
#' @seealso
#' + <https://keras.io/api/layers/activations#relu-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/activations/relu>
#' @tether keras.activations.relu
activation_relu <-
function (x, negative_slope = 0, max_value = NULL,
    threshold = 0)
{
    args <- capture_args()
    do.call(keras$activations$relu, args)
}


#' Relu6 activation function.
#'
#' @description
#' It's the ReLU function, but truncated to a maximum value of 6.
#'
#' @param x
#' Input tensor.
#'
#' @export
#' @family activations
#' @seealso
#' + <https://keras.io/api/layers/activations#relu6-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/activations/relu6>
#' @tether keras.activations.relu6
activation_relu6 <-
function (x)
{
    args <- capture_args()
    do.call(keras$activations$relu6, args)
}


#' Scaled Exponential Linear Unit (SELU).
#'
#' @description
#' The Scaled Exponential Linear Unit (SELU) activation function is defined as:
#'
#' - `scale * x` if `x > 0`
#' - `scale * alpha * (exp(x) - 1)` if `x < 0`
#'
#' where `alpha` and `scale` are pre-defined constants
#' (`alpha = 1.67326324` and `scale = 1.05070098`).
#'
#' Basically, the SELU activation function multiplies `scale` (> 1) with the
#' output of the `activation_elu` function to ensure a slope larger
#' than one for positive inputs.
#'
#' The values of `alpha` and `scale` are
#' chosen so that the mean and variance of the inputs are preserved
#' between two consecutive layers as long as the weights are initialized
#' correctly (see [`initializer_lecun_normal()`])
#' and the number of input units is "large enough"
#' (see reference paper for more information).
#'
#' # Notes
#' - To be used together with
#'   [`initializer_lecun_normal()`].
#' - To be used together with the dropout variant
#'     `layer_alpha_dropout()` (legacy, depracated).
#'
#' # Reference
#' - [Klambauer et al., 2017](https://arxiv.org/abs/1706.02515)
#'
#' @param x
#' Input tensor.
#'
#' @export
#' @family activations
#' @seealso
#' + <https://keras.io/api/layers/activations#selu-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/activations/selu>
#' @tether keras.activations.selu
activation_selu <-
function (x)
{
    args <- capture_args()
    do.call(keras$activations$selu, args)
}


#' Sigmoid activation function.
#'
#' @description
#' It is defined as: `sigmoid(x) = 1 / (1 + exp(-x))`.
#'
#' For small values (<-5),
#' `sigmoid` returns a value close to zero, and for large values (>5)
#' the result of the function gets close to 1.
#'
#' Sigmoid is equivalent to a 2-element softmax, where the second element is
#' assumed to be zero. The sigmoid function always returns a value between
#' 0 and 1.
#'
#' @param x
#' Input tensor.
#'
#' @export
#' @family activations
#' @seealso
#' + <https://keras.io/api/layers/activations#sigmoid-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/activations/sigmoid>
#' @tether keras.activations.sigmoid
activation_sigmoid <-
function (x)
{
    args <- capture_args()
    do.call(keras$activations$sigmoid, args)
}


#' Swish (or Silu) activation function.
#'
#' @description
#' It is defined as: `swish(x) = x * sigmoid(x)`.
#'
#' The Swish (or Silu) activation function is a smooth,
#' non-monotonic function that is unbounded above and
#' bounded below.
#'
#' # Reference
#' - [Ramachandran et al., 2017](https://arxiv.org/abs/1710.05941)
#'
#' @param x
#' Input tensor.
#'
#' @export
#' @family activations
#' @seealso
#' + <https://keras.io/api/layers/activations#silu-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/activations/silu>
#' @tether keras.activations.silu
activation_silu <-
function (x)
{
    args <- capture_args()
    do.call(keras$activations$silu, args)
}


#' Softmax converts a vector of values to a probability distribution.
#'
#' @description
#' The elements of the output vector are in range `[0, 1]` and sum to 1.
#'
#' Each input vector is handled independently.
#' The `axis` argument sets which axis of the input the function
#' is applied along.
#'
#' Softmax is often used as the activation for the last
#' layer of a classification network because the result could be interpreted as
#' a probability distribution.
#'
#' The softmax of each vector x is computed as
#' `exp(x) / sum(exp(x))`.
#'
#' The input values in are the log-odds of the resulting probability.
#'
#' @param x
#' Input tensor.
#'
#' @param axis
#' Integer, axis along which the softmax is applied.
#'
#' @export
#' @family activations
#' @seealso
#' + <https://keras.io/api/layers/activations#softmax-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/activations/softmax>
#' @tether keras.activations.softmax
activation_softmax <-
function (x, axis = -1L)
{
    args <- capture_args(list(axis = as_axis))
    do.call(keras$activations$softmax, args)
}


#' Softplus activation function.
#'
#' @description
#' It is defined as: `softplus(x) = log(exp(x) + 1)`.
#'
#' @param x
#' Input tensor.
#'
#' @export
#' @family activations
#' @seealso
#' + <https://keras.io/api/layers/activations#softplus-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/activations/softplus>
#' @tether keras.activations.softplus
activation_softplus <-
function (x)
{
    args <- capture_args()
    do.call(keras$activations$softplus, args)
}


#' Softsign activation function.
#'
#' @description
#' Softsign is defined as: `softsign(x) = x / (abs(x) + 1)`.
#'
#' @param x
#' Input tensor.
#'
#' @export
#' @family activations
#' @seealso
#' + <https://keras.io/api/layers/activations#softsign-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/activations/softsign>
#' @tether keras.activations.softsign
activation_softsign <-
function (x)
{
    args <- capture_args()
    do.call(keras$activations$softsign, args)
}


#' Hyperbolic tangent activation function.
#'
#' @description
#' It is defined as:
#' `tanh(x) = sinh(x) / cosh(x)`, i.e.
#' `tanh(x) = ((exp(x) - exp(-x)) / (exp(x) + exp(-x)))`.
#'
#' @param x
#' Input tensor.
#'
#' @export
#' @family activations
#' @seealso
#' + <https://keras.io/api/layers/activations#tanh-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/activations/tanh>
#' @tether keras.activations.tanh
activation_tanh <-
function (x)
{
    args <- capture_args()
    do.call(keras$activations$tanh, args)
}


#' Hard SiLU activation function, also known as Hard Swish.
#'
#' @description
#' It is defined as:
#'
#' - `0` if `if x < -3`
#' - `x` if `x > 3`
#' - `x * (x + 3) / 6` if `-3 <= x <= 3`
#'
#' It's a faster, piecewise linear approximation of the silu activation.
#'
#' # Reference
#' - [A Howard, 2019](https://arxiv.org/abs/1905.02244)
#'
#' @param x
#' Input tensor.
#'
#' @export
#' @tether keras.activations.hard_silu
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/activations/hard_silu>
activation_hard_silu <-
  structure(function (x)
  {
    args <- capture_args(NULL)
    do.call(keras$activations$hard_silu, args)
  }, py_function_name = "hard_silu")

#' @rdname activation_hard_silu
#' @export
activation_hard_swish <-
  structure(function (x)
  {
    args <- capture_args(NULL)
    do.call(keras$activations$hard_swish, args)
  }, py_function_name = "hard_silu")
