

#' Activation functions
#' 
#' Activations functions can either be used through [layer_activation()], or
#' through the activation argument supported by all forward layers.
#' 
#' @param x Input
#' @param alpha Alpha
#' @param max_value Maximum value
#'   
#' @rdname activation-functions
#' @export
activation_relu <- function(x, alpha = 0.0, max_value = NULL) {
  keras$activations$relu(
    x = x,
    alpha = alpha,
    max_value = max_value
  )
}

#' @rdname activation-functions
#' @export
activation_elu <- function(x, alpha = 1.0) {
  keras$activations$elu(
    x = x,
    alpha = alpha
  )
}

#' @rdname activation-functions
#' @export
activation_hard_sigmoid <- function(x) {
  keras$activations$hard_sigmoid(
    x = x
  )
}

#' @rdname activation-functions
#' @export
activation_linear <- function(x) {
  keras$activations$linear(
    x = x
  )
}

#' @rdname activation-functions
#' @export
activation_sigmoid <- function(x) {
  keras$activations$sigmoid(
    x = x
  )
}

#' @rdname activation-functions
#' @export
activation_softmax <- function(x) {
  keras$activations$softmax(
    x = x
  )
}

#' @rdname activation-functions
#' @export
activation_softplus <- function(x) {
  keras$activations$softplus(
    x = x
  )
}

#' @rdname activation-functions
#' @export
activation_softsign <- function(x) {
  keras$activations$softsign(
    x = x
  )
}

#' @rdname activation-functions
#' @export
activation_tanh <- function(x) {
  keras$activations$tanh(
    x = x
  )
}










