


#' @export
model <- function(input, output) {
  keras$models$Model(input = input, output = output)
}

#' @export
model_sequential <- function(layers = NULL, name = NULL) {
  keras$models$Sequential(layers = layers, name = name)
}

#' Configure a model for training
#' 
#' @param optimizer: Name of optimizer or optimizer object.
#' @param loss Name of objective function or objective function. If the model 
#'   has multiple outputs, you can use a different loss on each output by 
#'   passing a dictionary or a list of objectives.
#' @param metrics List of metrics to be evaluated by the model during training 
#'   and testing. Typically you will use `metrics='accuracy'`. To specify 
#'   different metrics for different outputs of a multi-output model, you could 
#'   also pass a named list such as `metrics=list(output_a = 'accuracy')`.
#' @param sample_weight_mode If you need to do timestep-wise sample weighting 
#'   (2D weights), set this to "temporal". `NULL` defaults to sample-wise
#'   weights (1D). If the model has multiple outputs, you can use a different 
#'   `sample_weight_mode` on each output by passing a list of modes.
#'   
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
fit <- function(model, x, y, batch_size = 32, nb_epoch = 10) {
  model$fit(
    x = x,
    y = y,
    nb_epoch = as.integer(nb_epoch),
    batch_size = as.integer(batch_size)
  )
  model
}


#' Save a model into a single HDF5 file
#' 
#' @param model Model to save
#' @param filepath File path to save to
#' @param overwrite Overwrite existing file if necessary
#' 
#' @details The following components of the model are saved: 
#' 
#'   - The model architecture, allowing to re-instantiate the model. 
#'   - The model weights. 
#'   - The state of the optimizer, allowing to resume training exactly where you
#'     left off.
#' This allows you to save the entirety of the state of a model
#' in a single file.
#' 
#' Saved models can be reinstantiated via [load_model()]. The model returned by
#' `load_model` is a compiled model ready to be used (unless the saved model
#' was never compiled in the first place).
#' 
#' @seealso [load_model()]
#' 
#' @export
save_model <- function(model, filepath, overwrite = TRUE) {
 keras$models$save_model(model = model, filepath = filepath, overwrite = overwrite) 
}


#' Load a model from an HDF5 file
#' 
#' @param filepath File path to load file from
#' @param custom_objects Napping class names (or function names) of custom 
#'   (non-Keras) objects to class/functions
#'   
#' @seealso [save_model()]   
#'   
#' @export
load_model <- function(filepath, custom_objects = NULL) {
  keras$models$load_model(filepath = filepath, custom_objects = custom_objects)
}



#' Predict Method for Keras Models
#' 
#' Generates output predictions for the input samples, processing the samples in
#' a batched way.
#' 
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
  if (is_null_xptr(object) || !py_available())
    cat("<pointer: 0x0>\n")
  else
    cat("Model\n", py_capture_output(object$summary(), type = "stdout"), sep="")
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



