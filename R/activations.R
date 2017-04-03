

#' Activation functions
#' 
#' Activations functions can either be used through [layer_activation()], or
#' through the activation argument supported by all forward layers.
#' 
#' @export
activation_relu <- function() {
  keras$activations$relu
}

#' @rdname activation_relu
#' @export
activation_elu <- function() {
  keras$activations$elu
}

#' @rdname activation_relu
#' @export
activation_hard_sigmoid <- function() {
  keras$activations$hard_sigmoid
}

#' @rdname activation_relu
#' @export
activation_linear <- function() {
  keras$activations$linear
}

#' @rdname activation_relu
#' @export
activation_sigmoid <- function() {
  keras$activations$sigmoid
}

#' @rdname activation_relu
#' @export
activation_softmax <- function() {
  keras$activations$softmax
}

#' @rdname activation_relu
#' @export
activation_softplus <- function() {
  keras$activations$softplus
}

#' @rdname activation_relu
#' @export
activation_softsign <- function() {
  keras$activations$softsign
}

#' @rdname activation_relu
#' @export
activation_tanh <- function() {
  keras$activations$tanh
}










