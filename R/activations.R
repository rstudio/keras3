

#' Activation functions
#' 
#' Activations functions can either be used through [layer_activation()], or
#' through the activation argument supported by all forward layers.
#' 
#' @param x Tensor
#' @param axis Integer, axis along which the softmax normalization is applied
#' @param alpha Alpha value
#' @param max_value Max value
#' 
#' @section References:
#' 
#'   - `activation_selu()`: [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)
#' 
#' @export
activation_relu <- function(x, alpha = 0.0, max_value = NULL) {
  keras$activations$relu(x, alpha = alpha, max_value = max_value)
}
attr(activation_relu, "py_function_name") <- "relu"


#' @rdname activation_relu
#' @export
activation_elu <- function(x, alpha = 1.0) {
  keras$activations$elu(x, alpha = alpha)
}
attr(activation_elu, "py_function_name") <- "elu"


#' @rdname activation_relu
#' @export
activation_selu <- function(x, alpha = 1.0) {
  keras$activations$selu(x)
}
attr(activation_elu, "py_function_name") <- "selu"


#' @rdname activation_relu
#' @export
activation_hard_sigmoid <- function(x) {
  keras$activations$hard_sigmoid(x)
}
attr(activation_hard_sigmoid, "py_function_name") <- "hard_sigmoid"

#' @rdname activation_relu
#' @export
activation_linear <- function(x) {
  keras$activations$linear(x)
}
attr(activation_linear, "py_function_name") <- "linear"

#' @rdname activation_relu
#' @export
activation_sigmoid <- function(x) {
  keras$activations$softmax(x)
}
attr(activation_sigmoid, "py_function_name") <- "sigmoid"

#' @rdname activation_relu
#' @export
activation_softmax <- function(x, axis = -1) {
  args <- list(x = x)
  if (keras_version() >= "2.0.2")
    args$axis <- as.integer(axis)
  do.call(keras$activations$softmax, args)
}
attr(activation_softmax, "py_function_name") <- "softmax"

#' @rdname activation_relu
#' @export
activation_softplus <- function(x) {
  keras$activations$softplus(x)
}
attr(activation_softplus, "py_function_name") <- "softplus"

#' @rdname activation_relu
#' @export
activation_softsign <- function(x) {
  keras$activations$softsign(x)
}
attr(activation_softsign, "py_function_name") <- "softsign"

#' @rdname activation_relu
#' @export
activation_tanh <- function(x) {
  keras$activations$tanh(x)
}
attr(activation_tanh, "py_function_name") <- "tanh"










