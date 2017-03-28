

#' Activation functions
#' 
#' Activations functions can either be used through [layer_activation()], or
#' through the activation argument supported by all forward layers.
#'
#' @name activation-functions
NULL

#' @rdname activation-functions
#' @export
activation_relu <- function() {
  keras$activations$relu
}

#' @rdname activation-functions
#' @export
activation_elu <- function() {
  keras$activations$elu
}

#' @rdname activation-functions
#' @export
activation_hard_sigmoid <- function() {
  keras$activations$hard_sigmoid
}

#' @rdname activation-functions
#' @export
activation_linear <- function() {
  keras$activations$linear
}

#' @rdname activation-functions
#' @export
activation_sigmoid <- function() {
  keras$activations$sigmoid
}

#' @rdname activation-functions
#' @export
activation_softmax <- function() {
  keras$activations$softmax
}

#' @rdname activation-functions
#' @export
activation_softplus <- function() {
  keras$activations$softplus
}

#' @rdname activation-functions
#' @export
activation_softsign <- function() {
  keras$activations$softsign
}

#' @rdname activation-functions
#' @export
activation_tanh <- function() {
  keras$activations$tanh
}










