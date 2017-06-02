
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
keras_model <- function(inputs, outputs = NULL) {
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
keras_model_sequential <- function(layers = NULL, name = NULL) {
  keras$models$Sequential(layers = layers, name = name)
}


#' Configure a Keras model for training
#' 
#' @param object Model object to compile.
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
compile <- function(object, optimizer, loss, metrics = NULL, loss_weights = NULL,
                    sample_weight_mode = NULL) {
  
  # ensure we are dealing with a list of metrics
  if (length(metrics) == 1)
    metrics <- list(metrics)
  
  # compile model
  object$compile(
    optimizer = optimizer, 
    loss = loss,
    metrics = metrics,
    loss_weights = loss_weights,
    sample_weight_mode = sample_weight_mode
  )
  
  # return model invisible (conventience for chaining)
  invisible(object)
}


#' Train a Keras model
#' 
#' Trains the model for a fixed number of epochs (iterations on a dataset).
#'
#' @param object Model to train.
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
#' @param ... Unused
#' 
#' @family model functions
#' 
#' @export
fit <- function(object, x, y, batch_size=32, epochs=10, verbose=1, callbacks=NULL,
                validation_split=0.0, validation_data=NULL, shuffle=TRUE,
                class_weight=NULL, sample_weight=NULL, initial_epoch=0, ...) {
  
  # fit the model
  history <- object$fit(
    x = normalize_x(x),
    y = normalize_x(y),
    batch_size = as.integer(batch_size),
    epochs = as.integer(epochs),
    verbose = as.integer(verbose),
    callbacks = normalize_callbacks(callbacks),
    validation_split = validation_split,
    validation_data = validation_data,
    shuffle = shuffle,
    class_weight = as_class_weight(class_weight),
    sample_weight = sample_weight,
    initial_epoch = as.integer(initial_epoch)
  )
  
  # return the history invisibly
  invisible(history)
}


#' Evaluate a Keras model

#' @inheritParams fit
#'   
#' @param object Model object to evaluate
#'   
#' @return Scalar test loss (if the model has a single output and no metrics) or
#'   list of scalars (if the model has multiple outputs and/or metrics).
#'   
#' @family model functions
#'   
#' @export
evaluate <- function(object, x, y, batch_size = 32, verbose=1, sample_weight = NULL) {
  object$evaluate(
    x = x,
    y = y,
    batch_size = as.integer(batch_size),
    verbose = as.integer(verbose),
    sample_weight = sample_weight
  )
}


#' Generate predictions from a Keras model
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
#' @family model functions
#' 
#' 
#' @importFrom stats predict
#' @export
predict.tensorflow.keras.engine.training.Model <- function(object, x, batch_size=32, verbose=0, ...) {
  
  # call predict
  object$predict(
    normalize_x(x), 
    batch_size = as.integer(batch_size),
    verbose = as.integer(verbose)
  )
}


#' Generates probability or class probability predictions for the input samples.
#' 
#' @inheritParams predict.tensorflow.keras.engine.training.Model
#' 
#' @param object Keras model object
#' 
#' @details The input samples are processed batch by batch.
#' 
#' @family model functions
#' 
#' @export
predict_proba <- function(object, x, batch_size = 32, verbose = 0) {
  object$predict_proba(
    x = normalize_x(x),
    batch_size = as.integer(batch_size),
    verbose = as.integer(verbose)
  )
}

#' @rdname predict_proba
#' @export
predict_classes <- function(object, x, batch_size = 32, verbose = 0) {
  object$predict_classes(
    x = normalize_x(x),
    batch_size = as.integer(batch_size),
    verbose = as.integer(verbose)
  )
}


#' Returns predictions for a single batch of samples.
#' 
#' @inheritParams predict.tensorflow.keras.engine.training.Model
#' 
#' @param object Keras model object
#' 
#' @return array of predictions.
#' 
#' @family model functions
#' 
#' @export
predict_on_batch <- function(object, x) {
  object$predict_on_batch(
    x = normalize_x(x)
  )
}


#' Single gradient update or model evaluation over one batch of samples.
#' 
#' @param object Keras model object
#' @param x input data, as an array or list of arrays (if the model has multiple
#'   inputs).
#' @param y labels, as an array.
#' @param class_weight named list mapping classes to a weight value, used for
#'   scaling the loss function (during training only).
#' @param sample_weight sample weights, as an array.
#'   
#' @return Scalar training or test loss (if the model has no metrics) or list of scalars
#' (if the model computes other metrics). The property `model$metrics_names`
#' will give you the display labels for the scalar outputs.
#' 
#' @family model functions
#'   
#' @export
train_on_batch <- function(object, x, y, class_weight = NULL, sample_weight = NULL) {
  object$train_on_batch(
    x = x,
    y = y,
    class_weight = as_class_weight(class_weight),
    sample_weight = sample_weight
  )
}

#' @rdname train_on_batch 
#' @export
test_on_batch <- function(object, x, y, sample_weight = NULL) {
  object$test_on_batch(
    x = x,
    y = y,
    sample_weight = sample_weight
  )
}



#' Fits the model on data yielded batch-by-batch by a generator.
#' 
#' The generator is run in parallel to the model, for efficiency. For instance,
#' this allows you to do real-time data augmentation on images on CPU in
#' parallel to training your model on GPU.
#' 
#' @param object Keras model object
#' @param generator a generator. The output of the generator must be either - a
#'   list (inputs, targets) - a list (inputs, targets, sample_weights). All
#'   arrays should contain the same number of samples. The generator is expected
#'   to loop over its data indefinitely. An epoch finishes when
#'   `steps_per_epoch` samples have been seen by the model.
#' @param steps_per_epoch Total number of steps (batches of samples) to yield
#'   from `generator` before declaring one epoch finished and starting the next
#'   epoch. It should typically be equal to the number of unique samples if your
#'   dataset divided by the batch size.
#' @param epochs integer, total number of iterations on the data.
#' @param verbose verbosity mode, 0, 1, or 2.
#' @param callbacks list of callbacks to be called during training.
#' @param validation_data this can be either - a generator for the validation
#'   data - a list (inputs, targets) - a list (inputs, targets, sample_weights).
#' @param validation_steps Only relevant if `validation_data` is a generator.
#'   Total number of steps (batches of samples) to yield from `generator` before
#'   stopping.
#' @param class_weight dictionary mapping class indices to a weight for the
#'   class.
#' @param max_q_size maximum size for the generator queue
#' @param workers maximum number of processes to spin up when using process
#'   based threading
#' @param pickle_safe if TRUE, use process based threading. Note that because
#'   this implementation relies on multiprocessing, you should not pass non
#'   picklable arguments to the generator as they can't be passed easily to
#'   children processes.
#' @param initial_epoch epoch at which to start training (useful for resuming a
#'   previous training run)
#'   
#'   
#' @return Training history object (invisibly)
#'   
#' @family model functions
#'   
#' @export
fit_generator <- function(object, generator, steps_per_epoch, epochs = 1, verbose = 1, 
                          callbacks = NULL, validation_data = NULL, validation_steps = NULL, 
                          class_weight = NULL, max_q_size = 10, workers = 1, 
                          pickle_safe = FALSE, initial_epoch = 0) {
  object$fit_generator(
    generator = generator,
    steps_per_epoch = as.integer(steps_per_epoch),
    epochs = as.integer(epochs),
    verbose = as.integer(verbose),
    callbacks = normalize_callbacks(callbacks),
    validation_data = validation_data,
    validation_steps = as_nullable_integer(validation_steps),
    class_weight = as_class_weight(class_weight),
    max_q_size = as.integer(max_q_size),
    workers = as.integer(workers),
    pickle_safe = pickle_safe,
    initial_epoch = as.integer(initial_epoch) 
  )
}

#' Evaluates the model on a data generator.
#' 
#' The generator should return the same kind of data as accepted by
#' `test_on_batch()`.
#' 
#' @inheritParams evaluate
#' 
#' @param generator Generator yielding lists (inputs, targets) or (inputs,
#'   targets, sample_weights)
#' @param steps Total number of steps (batches of samples) to yield from
#'   `generator` before stopping.
#' @param max_q_size maximum size for the generator queue
#' @param workers maximum number of processes to spin up when using process
#'   based threading
#' @param pickle_safe if `TRUE`, use process based threading. Note that because
#'   this implementation relies on multiprocessing, you should not pass non
#'   picklable arguments to the generator as they can't be passed easily to
#'   children processes.
#'   
#' @return Scalar test loss (if the model has a single output and no metrics) or
#'   list of scalars (if the model has multiple outputs and/or metrics). The
#'   attribute `model$metrics_names` will give you the display labels for the
#'   scalar outputs.
#'  
#' @family model functions   
#'     
#' @export
evaluate_generator <- function(object, generator, steps, max_q_size = 10, workers = 1, pickle_safe = FALSE) {
  object$evaluate_generator(
    generator = generator,
    steps = as.integer(steps),
    max_q_size = as.integer(max_q_size),
    workers = as.integer(workers),
    pickle_safe = pickle_safe
  )
}


#' Generates predictions for the input samples from a data generator.
#' 
#' The generator should return the same kind of data as accepted by 
#' `predict_on_batch()`.
#' 
#' @inheritParams predict.tensorflow.keras.engine.training.Model
#' 
#' @param object Keras model object
#' @param generator Generator yielding batches of input samples.
#' @param steps Total number of steps (batches of samples) to yield from
#'   `generator` before stopping.
#' @param max_q_size Maximum size for the generator queue.
#' @param workers Maximum number of processes to spin up when using process
#'   based threading
#' @param pickle_safe If `TRUE`, use process based threading. Note that because
#'   this implementation relies on multiprocessing, you should not pass non
#'   picklable arguments to the generator as they can't be passed easily to
#'   children processes.
#'   
#' @return Numpy array(s) of predictions.
#'   
#' @section Raises: ValueError: In case the generator yields data in an invalid
#'   format.
#'  
#' @family model functions   
#'     
#' @export
predict_generator <- function(object, generator, steps, max_q_size = 10, workers = 1, pickle_safe = FALSE) {
  object$predict_generator(
    generator = generator,
    steps = as.integer(steps),
    max_q_size = as.integer(max_q_size),
    workers = as.integer(workers),
    pickle_safe = pickle_safe
  )
}

  
#' Retrieves a layer based on either its name (unique) or index.
#' 
#' Indices are based on order of horizontal graph traversal (bottom-up) and 
#' are 0-based.
#' 
#' @param object Keras model object
#' @param name String, name of layer.
#' @param index Integer, index of layer (0-based)
#' 
#' @return A layer instance.
#' 
#' @family model functions   
#' 
#' @export
get_layer <- function(object, name = NULL, index = NULL) {
  object$get_layer(
    name = name,
    index = as_nullable_integer(index)
  )
}


#' Remove the last layer in a model
#' 
#' @param object Keras model object
#' 
#' @family model functions
#' 
#' @export
pop_layer <- function(object) {
  object$pop()
}


#' Print a summary of a Keras model
#' 
#' @param object Keras model instance
#' @param line_length Total length of printed lines
#' @param positions Relative or absolute positions of log elements in each line.
#'   If not provided, defaults to `c(0.33, 0.55, 0.67, 1.0)`.
#' @param ... Unused
#' 
#' @family model functions
#' 
#' @export
summary.tensorflow.keras.engine.training.Model <- function(object, line_length = getOption("width"), positions = NULL, ...) {
  if (py_is_null_xptr(object))
    cat("<pointer: 0x0>\n")
  else {
    cat(py_str(object, line_length = line_length, positions = positions), "\n")
  }
}

#' @importFrom reticulate py_str
#' @export
py_str.tensorflow.keras.engine.training.Model <- function(object,  line_length = getOption("width"), positions = NULL, ...) {
  paste0("Model\n", py_capture_output(object$summary(line_length = line_length, positions = positions), type = "stdout"))
}


#' Convert input data into a numpy array. This would be done 
#' automatically by reticulate for arrays and matrices however we
#' want to marshall arrays/matrices with C column ordering 
#' rather than the default Fortrain column ordering, as this will
#' make for more efficient copying of data to GPUs
normalize_x <- function(x) {
  
  # recurse for lists
  if (is.list(x))
    return(lapply(x, normalize_input))
  
  # convert to numpy
  if (!inherits(x, "numpy.ndarray")) {
    
    # convert non-array to array
    if (!is.array(x))
      x <- as.array(x)
    
    # do the conversion (will result in Fortran column ordering)
    x <- r_to_py(x)
  }
  
  # ensure we use C column ordering (won't create a new array if the array
  # is already using C ordering)
  x$astype(dtype = x$dtype, order = 'C', copy = FALSE)
}

as_class_weight <- function(class_weight) {
  # convert class weights to python dict
  if (!is.null(class_weight)) {
    if (is.list(class_weight))
      class_weight <- dict(class_weight)
    else
      stop("class_weight must be a named list of weights")
  }
}

have_module <- function(module) {
  tryCatch({ import(module); TRUE; }, error = function(e) FALSE)
}

have_h5py <- function() {
  have_module("h5py")
}

have_pyyaml <- function() {
  have_module("yaml")
}

have_requests <- function() {
  have_module("requests")
}

have_pillow <- function() {
  have_module("PIL") # aka Pillow
}

confirm_overwrite <- function(filepath, overwrite) {
  if (overwrite)
    TRUE 
  else {
    if (file.exists(filepath)) {
      if (interactive()) {
        prompt <- readline(sprintf("[WARNING] %s already exists - overwrite? [y/n] ", filepath))
        tolower(prompt) == 'y'
      } else {
        stop("File '", filepath, "' already exists (pass overwrite = TRUE to force save).", 
             call. = FALSE)
      }
    } else {
      TRUE
    }
  }
} 



