
#' Keras Model
#' 
#' A model is a directed acyclic graph of layers.
#' 
#' @param inputs Input layer
#' @param outputs Output layer
#'
#' @family model functions
#'
#' @export
model <- function(inputs, outputs) {
  keras$models$Model(inputs = inputs, outputs = outputs)
}


#' Keras Model composed of a linear stack of layers
#' 
#' @param layers List of layers to add to the model
#' @param name Name of model
#'   
#' @note
#' 
#' The first layer passed to a Sequential model should have a defined input
#' shape. What that means is that it should have received an `input_shape` or
#' `batch_input_shape` argument, or for some type of layers (recurrent,
#' Dense...) an `input_dim` argument.
#' 
#' @family model functions
#' 
#' @export
model_sequential <- function(layers = NULL, name = NULL) {
  keras$models$Sequential(layers = layers, name = name)
}

#' Configure a Keras model for training
#' 
#' @param model Model to compile.
#' @param optimizer Name of optimizer or optimizer object.
#' @param loss Name of objective function or objective function. If the model 
#'   has multiple outputs, you can use a different loss on each output by 
#'   passing a dictionary or a list of objectives.
#' @param metrics List of metrics to be evaluated by the model during training 
#'   and testing. Typically you will use `metrics='accuracy'`. To specify 
#'   different metrics for different outputs of a multi-output model, you could 
#'   also pass a named list such as `metrics=list(output_a = 'accuracy')`.
#' @param loss_weights Loss weights
#' @param sample_weight_mode If you need to do timestep-wise sample weighting 
#'   (2D weights), set this to "temporal". `NULL` defaults to sample-wise
#'   weights (1D). If the model has multiple outputs, you can use a different 
#'   `sample_weight_mode` on each output by passing a list of modes.
#'   
#' @family model functions
#'   
#' @export
compile <- function(model, optimizer, loss, metrics = NULL, loss_weights = NULL,
                    sample_weight_mode = NULL) {
  
  # resolve metrics (if they are functions in our namespace then call them 
  # so we end up passing the underlying python function not the R function)
  if (!is.null(metrics)) {
    # ensure we are dealing with a list
    if (is.function(metrics))
      metrics <- list(metrics)
    # resolve functions as necessary
    metrics <- lapply(metrics, resolve_keras_function)
  }
  
  # compile model
  model$compile(
    optimizer = optimizer, 
    loss = loss,
    metrics = metrics,
    loss_weights = loss_weights,
    sample_weight_mode = sample_weight_mode
  )
  
  # return it invisibly
  invisible(model)
}


#' Train a Keras model
#' 
#' Trains the model for a fixed number of epochs (iterations on a dataset).
#'
#' @param model Model to train.
#' @param x Vector, matrix, or array of training data (or list if the model has 
#'   multiple inputs). If all inputs in the model are named, you can also pass a
#'   list mapping input names to data.
#' @param y  Vector, matrix, or array of target data (or list if the model has 
#'   multiple outputs). If all outputs in the model are named, you can also pass
#'   a list mapping output names to data.
#' @param batch_size Number of samples per gradient update.
#' @param epochs Number of times to iterate over the training data arrays.
#' @param verbose  Verbosity mode (0 = silent, 1 = verbose, 2 = one log line per
#'   epoch).
#' @param callbacks List of callbacks to be called during training.
#' @param validation_split Float between 0 and 1: fraction of the training data 
#'   to be used as validation data. The model will set apart this fraction of 
#'   the training data, will not train on it, and will evaluate the loss and any
#'   model metrics on this data at the end of each epoch.
#' @param validation_data Data on which to evaluate the loss and any model 
#'   metrics at the end of each epoch. The model will not be trained on this 
#'   data. This could be a list (x_val, y_val) or a list (x_val, y_val, 
#'   val_sample_weights).
#' @param shuffle `TRUE` to shuffle the training data before each epoch.
#' @param class_weight Optional named list mapping indices (integers) to a
#'   weight (float) to apply to the model's loss for the samples from this class
#'   during training. This can be useful to tell the model to "pay more 
#'   attention" to samples from an under-represented class.
#' @param sample_weight Optional array of the same length as x, containing 
#'   weights to apply to the model's loss for each sample. In the case of 
#'   temporal data, you can pass a 2D array with shape (samples, 
#'   sequence_length), to apply a different weight to every timestep of every 
#'   sample. In this case you should make sure to specify 
#'   sample_weight_mode="temporal" in [compile()].
#' @param initial_epoch epoch at which to start training (useful for resuming a
#'   previous training run).
#' 
#' @family model functions
#' 
#' @export
fit <- function(model, x, y, batch_size=32, epochs=10, verbose=1, callbacks=NULL,
                validation_split=0.0, validation_data=NULL, shuffle=TRUE,
                class_weight=NULL, sample_weight=NULL, initial_epoch=0) {
  
  # convert class weights to python dict
  if (!is.null(class_weight)) {
    if (is.list(class_weight))
      class_weight <- dict(class_weight)
    else
      stop("class_weight must be a named list of weights")
  }
  
  # fit the model
  history <- model$fit(
    x = as.array(x),
    y = as.array(y),
    batch_size = as.integer(batch_size),
    epochs = as.integer(epochs),
    verbose = as.integer(verbose),
    callbacks = callbacks,
    validation_split = validation_split,
    validation_data = validation_data,
    shuffle = shuffle,
    class_weight = class_weight,
    sample_weight = sample_weight,
    initial_epoch = as.integer(initial_epoch)
  )
  
  # return the history invisibly
  invisible(history)
}


#' Evaluate a Keras model

#' @inheritParams fit
#'   
#' @return Scalar test loss (if the model has a single output and no metrics) or
#'   list of scalars (if the model has multiple outputs and/or metrics).
#'   
#' @family model functions
#'   
#' @export
evaluate <- function(model, x, y, batch_size = 32, verbose=1, sample_weight = NULL) {
  model$evaluate(
    x = x,
    y = y,
    batch_size = as.integer(batch_size),
    verbose = as.integer(verbose),
    sample_weight = sample_weight
  )
}

#' Save a Keras model into a single HDF5 file
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
#' @family model functions
#' 
#' @export
save_model <- function(model, filepath, overwrite = TRUE) {
  if (!have_h5py())
    stop("The h5py Python package is required to save and load models")
  keras$models$save_model(model = model, filepath = filepath, overwrite = overwrite)
}


#' Load a Keras model from an HDF5 file
#' 
#' @param filepath File path to load file from
#' @param custom_objects Mapping class names (or function names) of custom 
#'   (non-Keras) objects to class/functions
#'   
#' @family model functions   
#'   
#' @export
load_model <- function(filepath, custom_objects = NULL) {
  if (!have_h5py())
    stop("The h5py Python package is required to save and load models")
  keras$models$load_model(filepath = filepath, custom_objects = custom_objects)
}



#' Predict Method for Keras Models
#' 
#' Generates output predictions for the input samples, processing the samples in
#' a batched way.
#'
#' @param object Keras model
#' @param x Input data (vector, matrix, or array)
#' @param batch_size Integer
#' @param verbose Verbosity mode, 0 or 1.
#' @param ... Unused
#' 
#' @return vector, matrix, or array of predictions
#' 
#' @name predict
#' 
#' @family model functions
#' 
#' @importFrom stats predict
#' @export
predict.tensorflow.contrib.keras.python.keras.engine.training.Model <- function(object, x, batch_size=32, verbose=0, ...) {
  
  # call predict
  model <- object
  model$predict(
    as.array(x), 
    batch_size = as.integer(batch_size),
    verbose = as.integer(verbose)
  )
}

#' @export
summary.tensorflow.contrib.keras.python.keras.engine.training.Model <- function(object, ...) {
  if (py_is_null_xptr(object))
    cat("<pointer: 0x0>\n")
  else
    object$summary()
}

#' @importFrom reticulate py_str
#' @export
py_str.tensorflow.contrib.keras.python.keras.engine.training.Model <- function(object, ...) {
  cat("Model\n", py_capture_output(object$summary(), type = "stdout"), sep="")
}


have_h5py <- function() {
  tryCatch({ import("h5py"); TRUE; }, error = function(e) FALSE)
}



