
#' Keras Model
#' 
#' A model is a directed acyclic graph of layers.
#' 
#' @param inputs Input layer
#' @param outputs Output layer
#'
#' @family model functions
#'
#' @examples 
#' \dontrun{
#' library(keras)
#' 
#' # input layer
#' inputs <- layer_input(shape = c(784))
#' 
#' # outputs compose input + dense layers
#' predictions <- inputs %>%
#'   layer_dense(units = 64, activation = 'relu') %>% 
#'   layer_dense(units = 64, activation = 'relu') %>% 
#'   layer_dense(units = 10, activation = 'softmax')
#' 
#' # create and compile model
#' model <- keras_model(inputs = inputs, outputs = predictions)
#' model %>% compile(
#'   optimizer = 'rmsprop',
#'   loss = 'categorical_crossentropy',
#'   metrics = c('accuracy')
#' )
#' }
#' @export
keras_model <- function(inputs, outputs = NULL) {
  keras$models$Model(inputs = unname(inputs), outputs = unname(outputs))
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
#' @examples 
#' \dontrun{
#'  
#' library(keras)
#' 
#' model <- keras_model_sequential() 
#' model %>% 
#'   layer_dense(units = 32, input_shape = c(784)) %>% 
#'   layer_activation('relu') %>% 
#'   layer_dense(units = 10) %>% 
#'   layer_activation('softmax')
#' 
#' model %>% compile(
#'   optimizer = 'rmsprop',
#'   loss = 'categorical_crossentropy',
#'   metrics = c('accuracy')
#' )
#' }
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
#'   passing a dictionary or a list of objectives. The loss value that will be
#'   minimized by the model will then be the sum of all individual losses.
#' @param metrics List of metrics to be evaluated by the model during training
#'   and testing. Typically you will use `metrics='accuracy'`. To specify
#'   different metrics for different outputs of a multi-output model, you could
#'   also pass a named list such as `metrics=list(output_a = 'accuracy')`.
#' @param loss_weights Optional list specifying scalar coefficients to weight
#'   the loss contributions of different model outputs. The loss value that will
#'   be minimized by the model will then be the *weighted sum* of all indvidual
#'   losses, weighted by the `loss_weights` coefficients.
#' @param sample_weight_mode If you need to do timestep-wise sample weighting
#'   (2D weights), set this to "temporal". `NULL` defaults to sample-wise
#'   weights (1D). If the model has multiple outputs, you can use a different
#'   `sample_weight_mode` on each output by passing a list of modes.
#' @param ... Additional named arguments passed to `tf$Session$run`.
#'
#' @family model functions
#'
#' @export
compile <- function(object, optimizer, loss, metrics = NULL, loss_weights = NULL,
                    sample_weight_mode = NULL, ...) {
  
  # handle metrics
  if (!is.null(metrics)) {
    
    # get metric names (if any)
    metric_names <- names(metrics)
    if (is.null(metric_names))
      metric_names <- rep_len("", length(metrics))
    
    # convert metrics to a list (adding names to any custom functions)
    metrics <- lapply(1:length(metrics), function(i) {
      metric <- metrics[[i]]
      if (is.function(metric) && nzchar(metric_names[[i]]))
        attr(metric, "py_function_name") <- metric_names[[i]]
      metric
    })
  }
  
  # compile model
  object$compile(
    optimizer = optimizer, 
    loss = loss,
    metrics = metrics,
    loss_weights = loss_weights,
    sample_weight_mode = sample_weight_mode,
    ...
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
#' @param view_metrics View realtime plot of training metrics (by epoch). The
#'   default (`"auto"`) will display the plot when running within RStudio,
#'   `metrics` were specified during model [compile()], `epochs > 1` and
#'   `verbose > 0`. Use the global `keras.view_metrics` option to establish a
#'   different default.
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
fit <- function(object, x, y, batch_size=32, epochs=10, 
                verbose=1, callbacks=NULL,
                view_metrics = getOption("keras.view_metrics", default = "auto"),
                validation_split=0.0, validation_data=NULL, shuffle=TRUE,
                class_weight=NULL, sample_weight=NULL, initial_epoch=0, ...) {
  
  # resolve view_metrics
  if (identical(view_metrics, "auto"))
    view_metrics <- resolve_view_metrics(verbose, epochs, object$metrics)
  
  # fit the model
  history <- object$fit(
    x = to_numpy_array(x),
    y = to_numpy_array(y),
    batch_size = as.integer(batch_size),
    epochs = as.integer(epochs),
    verbose = as.integer(verbose),
    callbacks = normalize_callbacks(view_metrics, callbacks),
    validation_split = validation_split,
    validation_data = validation_data,
    shuffle = shuffle,
    class_weight = as_class_weight(class_weight),
    sample_weight = sample_weight,
    initial_epoch = as.integer(initial_epoch)
  )
  
  # turn history into an R object so it can be persited and
  # and give it a class so we can write print/plot methods
  params <- history$params
  if (params$do_validation)
    params$validation_samples <- dim(history$validation_data[[1]])[[1]]
  history <- keras_training_history(
    params = params,
    metrics = lapply(history$history, as.numeric)
  )
  
  # return the history invisibly
  invisible(history)
}


#' Evaluate a Keras model

#' @inheritParams fit
#'   
#' @param object Model object to evaluate
#'   
#' @return Named list of model test loss (or losses for models with multiple outputs) 
#'   and model metrics.
#'   
#' @family model functions
#'   
#' @export
evaluate <- function(object, x, y, batch_size = 32, verbose=1, sample_weight = NULL) {
  
  # perform evaluation
  result <- object$evaluate(
    x = to_numpy_array(x),
    y = to_numpy_array(y),
    batch_size = as.integer(batch_size),
    verbose = as.integer(verbose),
    sample_weight = sample_weight
  )
  
  # apply names
  names(result) <- object$metrics_names
  
  # write run data
  tfruns::write_run_metadata("evaluation", result)
  
  # return result
  result
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
predict.keras.engine.training.Model <- function(object, x, batch_size=32, verbose=0, ...) {
  
  # call predict
  object$predict(
    to_numpy_array(x), 
    batch_size = as.integer(batch_size),
    verbose = as.integer(verbose)
  )
}


#' Generates probability or class probability predictions for the input samples.
#' 
#' @inheritParams predict.keras.engine.training.Model
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
    x = to_numpy_array(x),
    batch_size = as.integer(batch_size),
    verbose = as.integer(verbose)
  )
}

#' @rdname predict_proba
#' @export
predict_classes <- function(object, x, batch_size = 32, verbose = 0) {
  object$predict_classes(
    x = to_numpy_array(x),
    batch_size = as.integer(batch_size),
    verbose = as.integer(verbose)
  )
}


#' Returns predictions for a single batch of samples.
#' 
#' @inheritParams predict.keras.engine.training.Model
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
    x = to_numpy_array(x)
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
    x = to_numpy_array(x),
    y = to_numpy_array(y),
    class_weight = as_class_weight(class_weight),
    sample_weight = sample_weight
  )
}

#' @rdname train_on_batch 
#' @export
test_on_batch <- function(object, x, y, sample_weight = NULL) {
  object$test_on_batch(
    x = to_numpy_array(x),
    y = to_numpy_array(y),
    sample_weight = sample_weight
  )
}



#' Fits the model on data yielded batch-by-batch by a generator.
#'
#' The generator is run in parallel to the model, for efficiency. For instance,
#' this allows you to do real-time data augmentation on images on CPU in
#' parallel to training your model on GPU.
#' 
#' @inheritParams fit 
#'
#' @param object Keras model object
#' @param generator A generator (e.g. like the one provided by
#'   [flow_images_from_directory()] or a custom R [generator function](https://rstudio.github.io/reticulate/articles/introduction.html#generators)).
#'
#'   The output of the generator must be a list of one of these forms:
#'      
#'      - (inputs, targets)
#'      - (inputs, targets, sample_weights)
#'      
#'   Note that the generator should call the [to_numpy_array()] function on its
#'   results prior to returning them (this ensures that arrays are provided in 
#'   'C' order and using the default floating point type for the backend.)
#'      
#'   All arrays should contain the same number of samples. The generator is expected
#'   to loop over its data indefinitely. An epoch finishes when `steps_per_epoch`
#'   batches have been seen by the model.
#' @param steps_per_epoch Total number of steps (batches of samples) to yield
#'   from `generator` before declaring one epoch finished and starting the next
#'   epoch. It should typically be equal to the number of unique samples if your
#'   dataset divided by the batch size.
#' @param epochs integer, total number of iterations on the data.
#' @param callbacks list of callbacks to be called during training.
#' @param validation_data this can be either: 
#'    - a generator for the validation data 
#'    - a list (inputs, targets) 
#'    - a list (inputs, targets, sample_weights).
#' @param validation_steps Only relevant if `validation_data` is a generator.
#'   Total number of steps (batches of samples) to yield from `generator` before
#'   stopping.
#' @param class_weight dictionary mapping class indices to a weight for the
#'   class.
#' @param max_queue_size maximum size for the generator queue
#' @param initial_epoch epoch at which to start training (useful for resuming a
#'   previous training run)
#'
#' @return Training history object (invisibly)
#'
#' @family model functions
#'
#' @export
fit_generator <- function(object, generator, steps_per_epoch, epochs = 1, 
                          verbose = 1, callbacks = NULL, 
                          view_metrics = getOption("keras.view_metrics", default = "auto"),
                          validation_data = NULL, validation_steps = NULL, 
                          class_weight = NULL, max_queue_size = 10, initial_epoch = 0) {
  
  # resolve view_metrics
  if (identical(view_metrics, "auto"))
    view_metrics <- resolve_view_metrics(verbose, epochs, object$metrics)
  
  call_generator_function(object$fit_generator, list(
    generator = as_generator(generator),
    steps_per_epoch = as.integer(steps_per_epoch),
    epochs = as.integer(epochs),
    verbose = as.integer(verbose),
    callbacks = normalize_callbacks(view_metrics, callbacks),
    validation_data = validation_data,
    validation_steps = as_nullable_integer(validation_steps),
    class_weight = as_class_weight(class_weight),
    max_queue_size = as.integer(max_queue_size),
    initial_epoch = as.integer(initial_epoch) 
  ))
}

#' Evaluates the model on a data generator.
#' 
#' The generator should return the same kind of data as accepted by
#' `test_on_batch()`.
#' 
#' @inheritParams evaluate
#' 
#' 
#' @param generator Generator yielding lists (inputs, targets) or (inputs,
#'   targets, sample_weights)
#' @param steps Total number of steps (batches of samples) to yield from
#'   `generator` before stopping.
#' @param max_queue_size maximum size for the generator queue
#'   
#' @return Named list of model test loss (or losses for models with multiple outputs) 
#'   and model metrics.
#'  
#' @family model functions   
#'     
#' @export
evaluate_generator <- function(object, generator, steps, max_queue_size = 10) {
  
  # perform evaluation
  result <- call_generator_function(object$evaluate_generator, list(
    generator = generator,
    steps = as.integer(steps),
    max_queue_size = as.integer(max_queue_size)
  ))
  
  # apply names
  names(result) <- object$metrics_names
  
  # write run data
  tfruns::write_run_metadata("evaluation", result)
  
  # return result
  result
}


#' Generates predictions for the input samples from a data generator.
#' 
#' The generator should return the same kind of data as accepted by 
#' `predict_on_batch()`.
#' 
#' @inheritParams predict.keras.engine.training.Model
#' 
#' @param object Keras model object
#' @param generator Generator yielding batches of input samples.
#' @param steps Total number of steps (batches of samples) to yield from
#'   `generator` before stopping.
#' @param max_queue_size Maximum size for the generator queue.
#' @param verbose verbosity mode, 0 or 1.
#'   
#' @return Numpy array(s) of predictions.
#'   
#' @section Raises: ValueError: In case the generator yields data in an invalid
#'   format.
#'  
#' @family model functions   
#'     
#' @export
predict_generator <- function(object, generator, steps, max_queue_size = 10, verbose = 0) {
  
  args <- list(
    generator = generator,
    steps = as.integer(steps),
    max_queue_size = as.integer(max_queue_size)
  )
  
  if (keras_version() >= "2.0.1")
    args$verbose <- as.integer(verbose)
  
  call_generator_function(object$predict_generator, args)
}


call_generator_function <- function(func, args) {
  
  # convert R function to Python iterator if necessary
  args$generator <- as_generator(args$generator)
  
  # force use of single background thread
  args$workers = 1L
  if (keras_version() >= "2.0.6")
    args$use_multiprocessing <- FALSE
  else {
    args$max_q_size <- args$max_queue_size
    args$max_queue_size <- NULL
    args$pickle_safe <- FALSE
  }
  
  # call the generator
  do.call(func, args)
}


as_generator <- function(x) {
  UseMethod("as_generator")
}

as_generator.default <- function(x) {
  stop("Unable to convert object to generator")
}

as_generator.python.builtin.object <- function(x) {
  x
}

as_generator.function <- function(x) {
  reticulate::py_iterator(x)
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
summary.keras.engine.training.Model <- function(object, line_length = getOption("width"), positions = NULL, ...) {
  if (py_is_null_xptr(object))
    cat("<pointer: 0x0>\n")
  else {
    if (keras_version() >= "2.0.6")
      object$summary(line_length = getOption("width"), print_fn = function(object) cat(object, "\n", sep = ""))
    else
      cat(py_str(object, line_length = line_length, positions = positions), "\n")
  }
}

#' @importFrom reticulate py_str
#' @export
py_str.keras.engine.training.Model <- function(object,  line_length = getOption("width"), positions = NULL, ...) {
  paste0("Model\n", py_capture_output(object$summary(line_length = line_length, positions = positions), type = "stdout"))
}


# determine whether to view metrics or not
resolve_view_metrics <- function(verbose, epochs, metrics) {
  (epochs > 1)          &&            # more than 1 epoch
  (length(metrics) > 0) &&            # capturing at least one metric
  (verbose > 0) &&                    # verbose mode is on
  !is.null(getOption("viewer")) &&    # have an internal viewer available
  nzchar(Sys.getenv("RSTUDIO"))       # running under RStudio
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



