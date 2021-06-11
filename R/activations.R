

#' Activation functions
#'
#' Activations functions can either be used through [layer_activation()], or
#' through the activation argument supported by all forward layers.
#'
#' @details
#'   - `activation_selu()` to be used together with the initialization "lecun_normal".
#'   - `activation_selu()` to be used together with the dropout variant "AlphaDropout".
#'
#' @param x Tensor
#' @param axis Integer, axis along which the softmax normalization is applied
#' @param alpha Alpha value
#' @param max_value Max value
#' @param threshold Threshold value for thresholded activation.
#'
#' @return Tensor with the same shape and dtype as \code{x}.
#'
#' @section References:
#'
#'   - `activation_swish()`: [Searching for Activation Functions](https://arxiv.org/abs/1710.05941)
#'   - `activation_gelu()`: [Gaussian Error Linear Units (GELUs)](https://arxiv.org/abs/1606.08415)
#'   - `activation_selu()`: [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)
#'   - `activation_elu()`: [Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)](https://arxiv.org/abs/1511.07289)
#'
#' @seealso <https://www.tensorflow.org/api_docs/python/tf/keras/activations>
#' @export
#' @description `relu(...)`: Applies the rectified linear unit activation function.
activation_relu <- function(x, alpha = 0.0, max_value = NULL, threshold = 0.0) {
  args <- list(
    x = x,
    alpha = alpha,
    max_value = max_value
  )
  if (keras_version() >= "2.2.3")
    args$threshold <- threshold

  do.call(keras$activations$relu, args)
}
attr(activation_relu, "py_function_name") <- "relu"


#' @rdname activation_relu
#' @description `elu(...)`: Exponential Linear Unit.
#' @export
activation_elu <- function(x, alpha = 1.0) {
  keras$activations$elu(x, alpha = alpha)
}
attr(activation_elu, "py_function_name") <- "elu"


#' @rdname activation_relu
#' @export
#' @description  `selu(...)`: Scaled Exponential Linear Unit (SELU).
activation_selu <- function(x) {
  keras$activations$selu(x)
}
attr(activation_selu, "py_function_name") <- "selu"


#' @rdname activation_relu
#' @export
#' @description `hard_sigmoid(...)`: Hard sigmoid activation function.
activation_hard_sigmoid <- function(x) {
  keras$activations$hard_sigmoid(x)
}
attr(activation_hard_sigmoid, "py_function_name") <- "hard_sigmoid"

#' @rdname activation_relu
#' @export
#' @description `linear(...)`: Linear activation function (pass-through).
activation_linear <- function(x) {
  keras$activations$linear(x)
}
attr(activation_linear, "py_function_name") <- "linear"

#' @rdname activation_relu
#' @export
#' @description `sigmoid(...)`: Sigmoid activation function, `sigmoid(x) = 1 / (1 + exp(-x))`.
activation_sigmoid <- function(x) {
  keras$activations$softmax(x)
}
attr(activation_sigmoid, "py_function_name") <- "sigmoid"

#' @rdname activation_relu
#' @export
#' @description `softmax(...)`: Softmax converts a vector of values to a probability distribution.
activation_softmax <- function(x, axis = -1) {
  args <- list(x = x)
  if (keras_version() >= "2.0.2")
    args$axis <- as.integer(axis)
  do.call(keras$activations$softmax, args)
}
attr(activation_softmax, "py_function_name") <- "softmax"

#' @rdname activation_relu
#' @export
#' @description `softplus(...)`: Softplus activation function, `softplus(x) = log(exp(x) + 1)`.
activation_softplus <- function(x) {
  keras$activations$softplus(x)
}
attr(activation_softplus, "py_function_name") <- "softplus"

#' @rdname activation_relu
#' @export
#' @description `softsign(...)`: Softsign activation function, `softsign(x) = x / (abs(x) + 1)`.
activation_softsign <- function(x) {
  keras$activations$softsign(x)
}
attr(activation_softsign, "py_function_name") <- "softsign"

#' @rdname activation_relu
#' @export
#' @description `tanh(...)`: Hyperbolic tangent activation function.
activation_tanh <- function(x) {
  keras$activations$tanh(x)
}
attr(activation_tanh, "py_function_name") <- "tanh"

#' @rdname activation_relu
#' @export
#' @description `exponential(...)`: Exponential activation function.
activation_exponential <- function(x) {
  keras$activations$exponential(x)
}
attr(activation_exponential, "py_function_name") <- "exponential"

#' @rdname activation_relu
#' @export
#' @description `gelu(...)`: Applies the Gaussian error linear unit (GELU) activation function.
#' @param approximate A bool, whether to enable approximation.
activation_gelu <- function(x, approximate=FALSE) {
  keras$activations$gelu(x, approximate)
}
attr(activation_gelu, "py_function_name") <- "gelu"

#' @rdname activation_relu
#' @export
#' @description `swish(...)`: Swish activation function, `swish(x) = x * sigmoid(x)`.
activation_swish <- function(x) {
  keras$activations$swish(x)
}
attr(activation_swish, "py_function_name") <- "swish"
