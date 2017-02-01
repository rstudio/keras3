

#' @export
layer_dense <- function(model, output_dim, input_dim = NULL) {
  model$add(kr$layers$Dense(output_dim, input_dim = input_dim))
  model
}

#' @export
layer_activation <- function(model, activation) {
  model$add(kr$layers$Activation(activation))
  model
}


