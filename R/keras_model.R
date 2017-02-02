


#' @export
model <- function(input, output) {
  kr$models$Model(input = input, output = output)
}

#' @export
sequential_model <- function(layers = NULL, name = NULL) {
  kr$models$Sequential(layers = layers, name = name)
}


#' @export
compile <- function(model, optimizer, loss, metrics = NULL, loss_weights = NULL,
                    sample_weight_mode = NULL) {
  model$compile(
    optimizer = optimizer, 
    loss = loss,
    metrics = ifelse(is.null(metrics), NULL, as.list(metrics)),
    loss_weights = loss_weights,
    sample_weight_mode = sample_weight_mode
  )
  model
}


#' @export
fit <- function(model, data, labels, batch_size = 32, nb_epoch = 10) {
  model$fit(
    data,
    labels,
    nb_epoch = as.integer(nb_epoch),
    batch_size = as.integer(batch_size)
  )
  model
}

#' @importFrom stats predict
#' @export
predict.keras.engine.training.Model <- function(object, x, batch_size=32, verbose=0) {
  object$predict(
    x, 
    batch_size = as.integer(batch_size),
    verbose = as.integer(verbose)
  )
}

