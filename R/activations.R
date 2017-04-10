

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
#' @export
activation_relu <- function(x, alpha = 0.0, max_value = NULL) {
  if (missing(x))
    keras$activations$relu
  else
    keras$activations$relu(x, alpha = alpha, max_value = max_value)
}

#' @rdname activation_relu
#' @export
activation_elu <- function(x, alpha = 1.0) {
  if (missing(x))
    keras$activations$elu
  else
    keras$activations$elu(x, alpha = alpha)
}

#' @rdname activation_relu
#' @export
activation_hard_sigmoid <- function(x) {
  if (missing(x))
    keras$activations$hard_sigmoid
  else
    keras$activations$hard_sigmoid(x)
}

#' @rdname activation_relu
#' @export
activation_linear <- function(x) {
  if (missing(x))
    keras$activations$linear
  else
    keras$activations$linear(x)
}

#' @rdname activation_relu
#' @export
activation_sigmoid <- function(x) {
  if (missing(x))
    keras$activations$sigmoid
  else
    keras$activations$softmax(x)
}

#' @rdname activation_relu
#' @export
activation_softmax <- function(x, axis = -1) {
  if (missing(x))
    keras$activations$softmax
  else
    keras$activations$softmax(x, axis = as.integer(axis))
}

#' @rdname activation_relu
#' @export
activation_softplus <- function(x) {
  if (missing(x))
    keras$activations$softplus
  else
    keras$activations$softplus(x)
}

#' @rdname activation_relu
#' @export
activation_softsign <- function(x) {
  if (missing(x))
    keras$activations$softsign
  else
    keras$activations$softsign(x)
}

#' @rdname activation_relu
#' @export
activation_tanh <- function(x) {
  if (missing(x))
    keras$activations$tanh
  else
    keras$activations$tanh(x)
}










