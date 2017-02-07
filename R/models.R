


#' @export
model <- function(input, output) {
  keras$models$Model(input = input, output = output)
}

#' @export
model_sequential <- function(layers = NULL, name = NULL) {
  keras$models$Sequential(layers = layers, name = name)
}


#' @export
compile <- function(model, optimizer, loss, metrics = NULL, loss_weights = NULL,
                    sample_weight_mode = NULL) {
  model <- clone_model_if_possible(model)
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
predict.keras.engine.training.Model <- function(object, x, batch_size=32, verbose=0, ...) {
  model <- object
  model$predict(
    x, 
    batch_size = as.integer(batch_size),
    verbose = as.integer(verbose)
  )
}

#' @export
summary.keras.engine.training.Model <- function(object, ...) {
  if (is_null_xptr(object))
    cat("<pointer: 0x0>\n")
  else
    object$summary()
}

#' @importFrom utils str
#' @export
str.keras.engine.training.Model <- function(object, ...) {
  if (is_null_xptr(object))
    cat("<pointer: 0x0>\n")
  else
    cat("Model\n", py_capture_stdout(object$summary()), sep="")
}

#' @export
print.keras.engine.training.Model <- function(x, ...) {
  str(x, ...)
}


# helper function which attempts to clone a model (we can only clone models that
# can save/read their config, which excludes models which have no layers --
# this likely just a bug that will be resolved later)
clone_model_if_possible <- function(model) {
  if (length(model$layers) > 0)
    keras$models$model_from_json(model$to_json())
  else if (inherits(model, "keras.models.Sequential"))
    model_sequential(name = model$name)
  else
    model
}



