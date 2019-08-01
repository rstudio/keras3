
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

#' Replicates a model on different GPUs.
#' 
#' @param model A Keras model instance. To avoid OOM errors,
#'   this model could have been built on CPU, for instance
#'    (see usage example below).
#' @param gpus `NULL` to use all available GPUs (default). Integer >= 2 or 
#'   list of integers, number of GPUs or list of GPU IDs on which to create 
#'   model replicas.
#' @param cpu_merge A boolean value to identify whether to force
#'   merging model weights under the scope of the CPU or not.
#' @param cpu_relocation A boolean value to identify whether to
#'   create the model's weights under the scope of the CPU.
#'   If the model is not defined under any preceding device
#'   scope, you can still rescue it by activating this option.   
#'  
#' @return  A Keras model object which can be used just like the initial
#'  `model` argument, but which distributes its workload on multiple GPUs.
#' 
#' @details 
#' Specifically, this function implements single-machine
#' multi-GPU data parallelism. It works in the following way:
#'   - Divide the model's input(s) into multiple sub-batches.
#'   - Apply a model copy on each sub-batch. Every model copy
#'     is executed on a dedicated GPU.
#'    - Concatenate the results (on CPU) into one big batch.
#'    
#' E.g. if your `batch_size` is 64 and you use `gpus=2`,
#' then we will divide the input into 2 sub-batches of 32 samples,
#' process each sub-batch on one GPU, then return the full
#' batch of 64 processed samples.
#' 
#' This induces quasi-linear speedup on up to 8 GPUs.
#' 
#' This function is only available with the TensorFlow backend
#' for the time being.
#'
#' @section Model Saving:
#' 
#' To save the multi-gpu model, use [save_model_hdf5()] or 
#' [save_model_weights_hdf5()] with the template model (the argument you 
#' passed to `multi_gpu_model`), rather than the model returned 
#' by `multi_gpu_model`.
#'
#' @examples \dontrun{
#' 
#' library(keras)
#' library(tensorflow)
#' 
#' num_samples <- 1000
#' height <- 224
#' width <- 224
#' num_classes <- 1000
#' 
#' # Instantiate the base model (or "template" model).
#' # We recommend doing this with under a CPU device scope,
#' # so that the model's weights are hosted on CPU memory.
#' # Otherwise they may end up hosted on a GPU, which would
#' # complicate weight sharing.
#' with(tf$device("/cpu:0"), {
#'   model <- application_xception(
#'     weights = NULL,
#'     input_shape = c(height, width, 3),
#'     classes = num_classes
#'   )
#' })
#' 
#' # Replicates the model on 8 GPUs.
#' # This assumes that your machine has 8 available GPUs.
#' parallel_model <- multi_gpu_model(model, gpus = 8)
#' parallel_model %>% compile(
#'   loss = "categorical_crossentropy",
#'   optimizer = "rmsprop"
#' )
#' 
#' # Generate dummy data.
#' x <- array(runif(num_samples * height * width*3), 
#'            dim = c(num_samples, height, width, 3))
#' y <- array(runif(num_samples * num_classes), 
#'            dim = c(num_samples, num_classes))
#' 
#' # This `fit` call will be distributed on 8 GPUs.
#' # Since the batch size is 256, each GPU will process 32 samples.
#' parallel_model %>% fit(x, y, epochs = 20, batch_size = 256)
#' 
#' # Save model via the template model (which shares the same weights):
#' model %>% save_model_hdf5("my_model.h5")
#' }
#'
#' @family model functions
#'
#' @export
multi_gpu_model <- function(model, gpus = NULL, cpu_merge = TRUE, cpu_relocation = FALSE) {
  
  if (is.null(gpus) && keras_version() < "2.1.4") {
    stop("You must provide an explicit gpus argument in Keras versions ",
         "prior to 2.1.4")
  }
  
  args <- list(
    model = model,
    gpus = as_nullable_integer(gpus)
  )
  
  if (keras_version() >= "2.1.6") {
    args$cpu_merge <- cpu_merge
    args$cpu_relocation <- cpu_relocation
  }
  
  do.call(keras$utils$multi_gpu_model, args)
}


#' @importFrom reticulate py_to_r_wrapper
#' @export
py_to_r_wrapper.keras.engine.training.Model <- function(x) {
  function(object) {
    compose_layer(object, x)
  }
}

#' @export
py_to_r_wrapper.kerastools.model.RModel <- function(x) {
  function(...) {
    x$call(...)
  }
}


#' Clone a model instance.
#'
#' Model cloning is similar to calling a model on new inputs, except that it
#' creates new layers (and thus new weights) instead of sharing the weights of
#' the existing layers.
#'
#' @param model Instance of Keras model (could be a functional model or a
#'   Sequential model).
#' @param input_tensors Optional list of input tensors to build the model upon.
#'   If not provided, placeholders will be created.
#'
#' @export
clone_model <- function(model, input_tensors = NULL) {
  keras$models$clone_model(
    model = model,
    input_tensors = input_tensors
  )
}


#' Configure a Keras model for training
#'
#' @param object Model object to compile.
#' @param optimizer Name of optimizer or optimizer instance.
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
#' @param target_tensors By default, Keras will create a placeholder for the
#'   model's target, which will be fed with the target data during
#'   training. If instead you would like to use your own
#'   target tensor (in turn, Keras will not expect external
#'   data for these targets at training time), you
#'   can specify them via the `target_tensors` argument. It should be
#'   a single tensor (for a single-output sequential model),
#' @param weighted_metrics List of metrics to be evaluated and weighted
#'   by sample_weight or class_weight during training and testing
#' @param ... When using the Theano/CNTK backends, these arguments
#'   are passed into K.function. When using the TensorFlow backend,
#'   these arguments are passed into `tf$Session()$run`.
#'
#' @family model functions
#'
#' @export
compile.keras.engine.training.Model <-
  function(object, optimizer, loss, 
           metrics = NULL, 
           loss_weights = NULL,
           sample_weight_mode = NULL, 
           weighted_metrics = NULL,
           target_tensors = NULL,
           ...) {
  
  # give losses a name
  loss_name <- deparse(substitute(loss))
  if (is.function(loss) && !inherits(loss, "python.builtin.object"))
    attr(loss, "py_function_name") <- loss_name
  
  # handle metrics
  if (!is.null(metrics)) {
    
    # convert metrics to list if it isn't one
    if (!is.list(metrics) && length(metrics) == 1)
      metrics <- list(metrics)
    
    # get metric names (if any)
    metric_names <- names(metrics)
    if (is.null(metric_names))
      metric_names <- rep_len("", length(metrics))

    # if all the metrics names are output names then leave them alone
    # (just convert to a list with no special processing)
    if (py_has_attr(object, "output_names") && all(metric_names %in% object$output_names)) {
      metrics <- as.list(metrics)
    } else {
      # convert metrics to a list (adding names to any custom functions)
      metrics <- lapply(1:length(metrics), function(i) {
        metric <- metrics[[i]]
        
        if (is.function(metric) && nzchar(metric_names[[i]])) {
          warning("Passing names for custom metrics is deprecated. Please use the ",
                  "custom_metric() function to define custom metrics.")
          attr(metric, "py_function_name") <- metric_names[[i]]
        }
        
        metric
      })
    }
  }
  
  # args
  args <- list(
    optimizer = optimizer, 
    loss = loss,
    metrics = metrics,
    loss_weights = loss_weights,
    sample_weight_mode = sample_weight_mode
  )
  
  # keras 2.07 args
  if (keras_version() >= "2.0.7") {
    # weighted metrics
    if (!is.null(weighted_metrics) && !is.list(weighted_metrics))
      weighted_metrics <- list(weighted_metrics)
    args$weighted_metrics <- weighted_metrics
    # target tensors
    if (!is.null(target_tensors) && !is.list(target_tensors))
      target_tensors <- list(target_tensors)
    args$target_tensors <- target_tensors
  }
  
  # var args
  var_args <- list(...)
  args <- append(args, var_args)
  
  # compile model
  do.call(object$compile, args)
  
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
#'   list mapping input names to data. `x` can be `NULL` (default) if feeding 
#'   from framework-native tensors (e.g. TensorFlow data tensors).
#' @param y  Vector, matrix, or array of target (label) data (or list if the model has
#'   multiple outputs). If all outputs in the model are named, you can also pass
#'   a list mapping output names to data. `y` can be `NULL` (default) if feeding 
#'   from framework-native tensors (e.g. TensorFlow data tensors).
#' @param batch_size Integer or `NULL`. Number of samples per gradient update.
#'   If unspecified, `batch_size` will default to 32.
#' @param epochs Number of epochs to train the model.
#'   Note that in conjunction with `initial_epoch`,
#'   `epochs` is to be understood as "final epoch". The model is
#'   not trained for a number of iterations given by `epochs`, but
#'   merely until the epoch of index `epochs` is reached.
#' @param verbose  Verbosity mode (0 = silent, 1 = progress bar, 2 = one line per
#'   epoch).
#' @param view_metrics View realtime plot of training metrics (by epoch). The
#'   default (`"auto"`) will display the plot when running within RStudio,
#'   `metrics` were specified during model [compile()], `epochs > 1` and
#'   `verbose > 0`. Use the global `keras.view_metrics` option to establish a
#'   different default.
#' @param callbacks List of callbacks to be called during training.
#' @param validation_split Float between 0 and 1. Fraction of the training data
#'   to be used as validation data. The model will set apart this fraction of
#'   the training data, will not train on it, and will evaluate the loss and any
#'   model metrics on this data at the end of each epoch. The validation data
#'   is selected from the last samples in the `x` and `y` data provided, 
#'   before shuffling.
#' @param validation_data Data on which to evaluate the loss and any model
#'   metrics at the end of each epoch. The model will not be trained on this
#'   data. This could be a list (x_val, y_val) or a list (x_val, y_val,
#'   val_sample_weights). `validation_data` will override `validation_split`.
#' @param shuffle shuffle: Logical (whether to shuffle the training data
#'    before each epoch) or string (for "batch"). "batch" is a special option
#'    for dealing with the limitations of HDF5 data; it shuffles in batch-sized
#'    chunks. Has no effect when `steps_per_epoch` is not `NULL`.
#' @param class_weight Optional named list mapping indices (integers) to a
#'   weight (float) value, used for weighting the loss function
#'   (during training only). This can be useful to tell the model to
#'   "pay more attention" to samples from an under-represented class.
#' @param sample_weight Optional array of the same length as x, containing
#'   weights to apply to the model's loss for each sample. In the case of
#'   temporal data, you can pass a 2D array with shape (samples,
#'   sequence_length), to apply a different weight to every timestep of every
#'   sample. In this case you should make sure to specify
#'   `sample_weight_mode="temporal"` in [compile()].
#' @param initial_epoch Integer, Epoch at which to start training (useful for
#'   resuming a previous training run).
#' @param steps_per_epoch Total number of steps (batches of samples) before
#'   declaring one epoch finished and starting the next epoch. When training
#'   with input tensors such as TensorFlow data tensors, the default `NULL` is
#'   equal to the number of samples in your dataset divided by the batch
#'   size, or 1 if that cannot be determined. 
#' @param  validation_steps Only relevant if `steps_per_epoch` is specified. 
#'   Total number of steps (batches of samples) to validate before stopping.
#' @param ... Unused
#'
#' @return A `history` object that contains all information collected
#'   during training.
#'
#' @family model functions
#'
#' @export
fit.keras.engine.training.Model <- 
  function(object, x = NULL, y = NULL, batch_size=NULL, epochs=10, 
           verbose=getOption("keras.fit_verbose", default = 1), callbacks=NULL,
           view_metrics = getOption("keras.view_metrics", default = "auto"),
           validation_split=0.0, validation_data=NULL, shuffle=TRUE,
           class_weight=NULL, sample_weight=NULL, initial_epoch=0,
           steps_per_epoch=NULL, validation_steps=NULL, ...) {
    
  # defaults
  if (is.null(batch_size) && is.null(steps_per_epoch) && 
      !is_tensorflow_dataset(x))
    batch_size <- 32L
  
  # resolve view_metrics
  if (identical(view_metrics, "auto"))
    view_metrics <- resolve_view_metrics(verbose, epochs, object$metrics)
  
  # build args
  args <- list(
    batch_size = as_nullable_integer(batch_size),
    epochs = as.integer(epochs),
    verbose = as.integer(verbose),
    callbacks = normalize_callbacks_with_metrics(view_metrics, callbacks),
    validation_split = validation_split,
    shuffle = shuffle,
    class_weight = as_class_weight(class_weight),
    sample_weight = keras_array(sample_weight),
    initial_epoch = as.integer(initial_epoch)
  )
  
  # resolve validation_Data (check for TF dataset)
  if (!is.null(validation_data)) {
    dataset <- resolve_tensorflow_dataset(validation_data)
    if (!is.null(dataset))
      args$validation_data <- dataset
    else
      args$validation_data <- keras_array(validation_data)  
  }
    
  # resolve x and y (check for TF dataset)
  dataset <- resolve_tensorflow_dataset(x)
  if (inherits(dataset, "tensorflow.python.data.ops.dataset_ops.DatasetV2")) {
    args$x <- dataset
  } else if (!is.null(dataset)) {
    args$x <- dataset[[1]]
    args$y <- dataset[[2]]
  } else {
    if (!is.null(x))
      args$x <- keras_array(x)
    if (!is.null(y))
      args$y <- keras_array(y) 
  }
  
  if (keras_version() >= "2.0.7") {
    args$steps_per_epoch <- as_nullable_integer(steps_per_epoch)
    args$validation_steps <- as_nullable_integer(validation_steps)
  }
  
  # fit the model
  history <- do.call(object$fit, args)
  
  # convert to a keras_training history object
  history <- to_keras_training_history(history)
  
  # write metadata contained in history
  write_history_metadata(history)
  
  # return the history invisibly
  invisible(history)
}

#' Evaluate a Keras model

#' @inheritParams fit.keras.engine.training.Model
#'
#' @param object Model object to evaluate
#' @param x Vector, matrix, or array of test data (or list if the model has
#'   multiple inputs). If all inputs in the model are named, you can also pass a
#'   list mapping input names to data. `x` can be `NULL` (default) if feeding 
#'   from framework-native tensors (e.g. TensorFlow data tensors).
#' @param y  Vector, matrix, or array of target (label) data (or list if the model has
#'   multiple outputs). If all outputs in the model are named, you can also pass
#'   a list mapping output names to data. `y` can be `NULL` (default) if feeding 
#'   from framework-native tensors (e.g. TensorFlow data tensors).
#' @param steps Total number of steps (batches of samples) before declaring the
#'   evaluation round finished. Ignored with the default value of `NULL`.
#' @param callbacks List of callbacks to apply during evaluation.
#' @param ... Unused   
#'   
#'   
#' @return Named list of model test loss (or losses for models with multiple
#'   outputs) and model metrics.
#'
#' @family model functions
#'
#' @export
evaluate.keras.engine.training.Model <- function(object, x = NULL, y = NULL, batch_size = NULL, 
                                                 verbose=1, sample_weight = NULL, steps = NULL, 
                                                 callbacks = NULL, ...) {
  
  # defaults
  if (is.null(batch_size) && is.null(steps) &&!is_tensorflow_dataset(x))
    batch_size <- 32L
  
  # args
  args <- list(
    batch_size = as_nullable_integer(batch_size),
    verbose = as.integer(verbose),
    sample_weight = sample_weight
  )
  
  args <- resolve_callbacks(args, callbacks)
  
  # resolve x and y (check for TF dataset)
  dataset <- resolve_tensorflow_dataset(x)
  if (inherits(dataset, "tensorflow.python.data.ops.dataset_ops.DatasetV2")) {
    args$x <- dataset
  } else if (!is.null(dataset)) {
    args$x <- dataset[[1]]
    args$y <- dataset[[2]] 
  } else {
    args$x <- keras_array(x)
    args$y <- keras_array(y) 
  }
  
  if (keras_version() >= "2.0.7")
    args$steps <- as_nullable_integer(steps)
  
  # perform evaluation
  result <- do.call(object$evaluate, args)
  
  # apply names
  names(result) <- object$metrics_names
  
  # write run data
  tfruns::write_run_metadata("evaluation", result)
  
  # return result
  result
}

resolve_callbacks <- function(args, callbacks) {
  if (get_keras_implementation() == "tensorflow" && tensorflow::tf_version() >= "2.0") {
    args <- append(args, list(callbacks = normalize_callbacks(callbacks)))
  } else if (!is.null(callbacks)) {
    warning("Prediction callbacks are only supported for TensorFlow ",
            "implementation of Keras. And tf_version() >= 2.0")
  }
  args
}

#' Generate predictions from a Keras model
#' 
#' Generates output predictions for the input samples, processing the samples in
#' a batched way.
#'
#' @inheritParams evaluate.keras.engine.training.Model
#'
#' @param object Keras model
#' @param x Input data (vector, matrix, or array)
#' @param batch_size Integer. If unspecified, it will default to 32.
#' @param verbose Verbosity mode, 0 or 1.
#' @param callbacks List of callbacks to apply during prediction. 
#' @param ... Unused
#' 
#' @return vector, matrix, or array of predictions
#' 
#' @family model functions
#' 
#' 
#' @importFrom stats predict
#' @export
predict.keras.engine.training.Model <- function(object, x, batch_size=NULL, verbose=0, steps=NULL, 
                                                callbacks = NULL,...) {
  
  # defaults
  if (is.null(batch_size) && is.null(steps) &&!is_tensorflow_dataset(x))
    batch_size <- 32L
  
  # args
  args <- list(
    batch_size = as_nullable_integer(batch_size),
    verbose = as.integer(verbose)
  )
  
  args <- resolve_callbacks(args, callbacks)
  
  # resolve x (check for TF dataset)
  dataset <- resolve_tensorflow_dataset(x)
  if (inherits(dataset, "tensorflow.python.data.ops.dataset_ops.DatasetV2")) {
    args$x <- dataset 
  } else if (!is.null(dataset)) {
    args$x <- dataset[[1]]
  } else {
    args$x <- keras_array(x)
  }
  
  if (keras_version() >= "2.0.7")
    args$steps <- as_nullable_integer(steps)
  
  # call predict
  do.call(object$predict, args)
}


#' Generates probability or class probability predictions for the input samples.
#' 
#' @inheritParams predict.keras.engine.training.Model
#' 
#' @param object Keras model object
#' @param steps Total number of steps (batches of samples) before declaring the
#'   evaluation round finished. The default `NULL` is equal to the number of 
#'   samples in your dataset divided by the batch size.
#'   
#' @details The input samples are processed batch by batch.
#' 
#' @family model functions
#' 
#' @export
predict_proba <- function(object, x, batch_size = NULL, verbose = 0, steps = NULL) {
  
  args <- list(
    batch_size = as_nullable_integer(batch_size),
    verbose = as.integer(verbose)
  )
  
  # resolve x (check for TF dataset)
  dataset <- resolve_tensorflow_dataset(x)
  if (!is.null(dataset)) {
    args$x <- dataset[[1]]
  } else {
    args$x <- keras_array(x)
  }
  
  if (keras_version() >= "2.1.3")
    args$steps <- as_nullable_integer(steps)
  
  do.call(object$predict_proba, args)
}

#' @rdname predict_proba
#' @export
predict_classes <- function(object, x, batch_size = NULL, verbose = 0, steps = NULL) {
  args <- list(
    batch_size = as_nullable_integer(batch_size),
    verbose = as.integer(verbose)
  )
  
  # resolve x (check for TF dataset)
  dataset <- resolve_tensorflow_dataset(x)
  if (!is.null(dataset)) {
    args$x <- dataset[[1]]
  } else {
    args$x <- keras_array(x)
  }
  
  if (keras_version() >= "2.1.3")
    args$steps <- as_nullable_integer(steps)
  
  do.call(object$predict_classes, args)
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
    x = keras_array(x)
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
    x = keras_array(x),
    y = keras_array(y),
    class_weight = as_class_weight(class_weight),
    sample_weight = sample_weight
  )
}

#' @rdname train_on_batch 
#' @export
test_on_batch <- function(object, x, y, sample_weight = NULL) {
  object$test_on_batch(
    x = keras_array(x),
    y = keras_array(y),
    sample_weight = sample_weight
  )
}



#' Fits the model on data yielded batch-by-batch by a generator.
#'
#' The generator is run in parallel to the model, for efficiency. For instance,
#' this allows you to do real-time data augmentation on images on CPU in
#' parallel to training your model on GPU.
#' 
#' @inheritParams fit.keras.engine.training.Model 
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
#'   This list (a single output of the generator) makes a single batch.
#'   Therefore, all arrays in this list must have the same length (equal to 
#'   the size of this batch). Different batches may have different sizes.
#'   For example, the last batch of the epoch is commonly smaller than the
#'   others, if the size of the dataset is not divisible by the batch size.
#'   The generator is expected to loop over its data indefinitely. An epoch
#'   finishes when `steps_per_epoch` batches have been seen by the model.
#' @param steps_per_epoch Total number of steps (batches of samples) to yield
#'   from `generator` before declaring one epoch finished and starting the next
#'   epoch. It should typically be equal to the number of samples if your
#'   dataset divided by the batch size.
#' @param epochs Integer. Number of epochs to train the model.
#'   An epoch is an iteration over the entire data provided, as defined by 
#'   `steps_per_epoch`. Note that in conjunction with `initial_epoch`,
#'   `epochs` is to be understood as "final epoch". The model is not trained
#'    for a number of iterations given by `epochs`, but merely until the epoch
#'    of index `epochs` is reached.
#' @param callbacks List of callbacks to apply during training.
#' @param validation_data this can be either: 
#'    - a generator for the validation data 
#'    - a list (inputs, targets) 
#'    - a list (inputs, targets, sample_weights).
#'  on which to evaluate
#'  the loss and any model metrics at the end of each epoch.
#'  The model will not be trained on this data.
#' @param validation_steps Only relevant if `validation_data` is a generator.
#'   Total number of steps (batches of samples) to yield from `generator` before
#'   stopping at the end of every epoch. It should typically be equal to the number
#'   of samples of your validation dataset divided by the batch size.
#' @param class_weight Optional named list mapping class indices (integer) to a
#'   weight (float) value, used for weighting the loss function (during 
#'   training only). This can be useful to tell the model to "pay more 
#'   attention" to samples from an under-represented class.
#' @param max_queue_size Maximum size for the generator queue. If unspecified,
#'   `max_queue_size` will default to 10.
#' @param workers Maximum number of threads to use for parallel processing. Note that
#'   parallel processing will only be performed for native Keras generators (e.g.
#'   `flow_images_from_directory()`) as R based generators must run on the main thread.
#' @param initial_epoch epoch at which to start training (useful for resuming a
#'   previous training run)
#'
#' @return Training history object (invisibly)
#'
#' @family model functions
#'
#' @export
fit_generator <- function(object, generator, steps_per_epoch, epochs = 1, 
                          verbose=getOption("keras.fit_verbose", default = 1), callbacks = NULL, 
                          view_metrics = getOption("keras.view_metrics", default = "auto"),
                          validation_data = NULL, validation_steps = NULL, 
                          class_weight = NULL, max_queue_size = 10, workers = 1, initial_epoch = 0) {
  
  # resolve view_metrics
  if (identical(view_metrics, "auto"))
    view_metrics <- resolve_view_metrics(verbose, epochs, object$metrics)
  
  if (is.list(validation_data))
    validation_data <- do.call(reticulate::tuple, keras_array(validation_data))
  
  history <- call_generator_function(object$fit_generator, list(
    generator = generator,
    steps_per_epoch = as.integer(steps_per_epoch),
    epochs = as.integer(epochs),
    verbose = as.integer(verbose),
    callbacks = normalize_callbacks_with_metrics(view_metrics, callbacks),
    validation_data = validation_data,
    validation_steps = as_nullable_integer(validation_steps),
    class_weight = as_class_weight(class_weight),
    max_queue_size = as.integer(max_queue_size),
    workers = as.integer(workers),
    initial_epoch = as.integer(initial_epoch) 
  ))
  
  # convert to a keras_training history object
  history <- to_keras_training_history(history)
  
  # write metadata from history
  write_history_metadata(history)
  
  # return the history invisibly
  invisible(history)
}

#' Evaluates the model on a data generator.
#' 
#' The generator should return the same kind of data as accepted by
#' `test_on_batch()`.
#' 
#' @inheritParams evaluate.keras.engine.training.Model
#' @inheritParams fit_generator
#' 
#' @param generator Generator yielding lists (inputs, targets) or (inputs,
#'   targets, sample_weights)
#' @param steps Total number of steps (batches of samples) to yield from
#'   `generator` before stopping.
#'   
#' @return Named list of model test loss (or losses for models with multiple outputs) 
#'   and model metrics.
#'  
#' @family model functions   
#'     
#' @export
evaluate_generator <- function(object, generator, steps, max_queue_size = 10, workers = 1,
                               callbacks = NULL) {
  
  args <- list(
    generator = generator,
    steps = as.integer(steps),
    max_queue_size = as.integer(max_queue_size),
    workers = as.integer(workers)
  )
  
  args <- resolve_callbacks(args, callbacks)
  
  # perform evaluation
  result <- call_generator_function(object$evaluate_generator, args)
  
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
#' @inheritParams fit_generator
#' 
#' @param object Keras model object
#' @param generator Generator yielding batches of input samples.
#' @param steps Total number of steps (batches of samples) to yield from
#'   `generator` before stopping.
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
predict_generator <- function(object, generator, steps, max_queue_size = 10, workers = 1, verbose = 0,
                              callbacks = NULL) {
  
  args <- list(
    generator = generator,
    steps = as.integer(steps),
    max_queue_size = as.integer(max_queue_size),
    workers = as.integer(workers)
  )
  
  if (keras_version() >= "2.0.1")
    args$verbose <- as.integer(verbose)
  
  args <- resolve_callbacks(args, callbacks)
  
  call_generator_function(object$predict_generator, args)
}


call_generator_function <- function(func, args) {
  
  # check if any generators should run on the main thread
  use_main_thread_generator <- 
    is_main_thread_generator(args$generator) ||
    is_main_thread_generator(args$validation_data)
  
  # handle generators
  args$generator <- as_generator(args$generator)
  if (!is.null(args$validation_data))
    args$validation_data <- as_generator(args$validation_data)
  
  # force use of thread based concurrency
  if (keras_version() >= "2.0.6")
    args$use_multiprocessing <- FALSE
  else {
    args$max_q_size <- args$max_queue_size
    args$max_queue_size <- NULL
    args$pickle_safe <- FALSE
  }
  
  # if it's a main thread generator then force workers to correct value
  if (use_main_thread_generator) {
    
    # error to use workers > 1 for main thread generator
    if (args$workers > 1) {
      stop('You may not specify workers > 1 for R based generator functions (R ',
           'generators must run on the main thread)', call. = FALSE)
    }
    
    # set workers to 0 for versions of keras that support this
    if (keras_version() >= "2.1.2")
      args$workers = 0L
    else
      args$workers = 1L
  }
  
  # call the generator
  do.call(func, args)
}


as_generator <- function(x) {
  UseMethod("as_generator")
}

as_generator.default <- function(x) {
  x
}

as_generator.tensorflow.python.data.ops.dataset_ops.Dataset <- function(x) {
  python_path <- system.file("python", package = "keras")
  tools <- reticulate::import_from_path("kerastools", path = python_path)
  tools$generator$dataset_generator(x , k_get_session())
}

as_generator.tensorflow.python.data.ops.dataset_ops.DatasetV2 <- function(x) {
   
  if (tensorflow::tf_version() >= "2.0")
    x
  else
    as_generator.tensorflow.python.data.ops.dataset_ops.Dataset(x)  
  
}
  
as_generator.function <- function(x) {
  python_path <- system.file("python", package = "keras")
  tools <- reticulate::import_from_path("kerastools", path = python_path)
  iter <- reticulate::py_iterator(function() {
    elem <- keras_array(x())
    
    # deals with the case where the generator is used for prediction and only
    # yields x's values.
    if (length(elem) == 1)
      elem[[2]] <- list()
    
    do.call(reticulate::tuple, elem)
  })
  tools$generator$iter_generator(iter)
}

as_generator.keras_preprocessing.sequence.TimeseriesGenerator <- function(x) {
  reticulate::as_iterator(x)
}

is_main_thread_generator <- function(x) {
  UseMethod("is_main_thread_generator")
}

is_main_thread_generator.default <- function(x) {
  FALSE
}

is_main_thread_generator.tensorflow.python.data.ops.dataset_ops.Dataset <- function(x) {
  TRUE
}

is_main_thread_generator.function <- function(x) {
  TRUE
}

is_main_thread_generator.keras.preprocessing.image.Iterator <- function(x) {
  if (py_has_attr(x, "image_data_generator")) {
    generator <- x$image_data_generator
    !is.null(generator$preprocessing_function)
  } else {
    FALSE
  }
}

is_main_thread_generator.keras_preprocessing.image.Iterator <- function(x) {
  if (py_has_attr(x, "image_data_generator")) {
    generator <- x$image_data_generator
    !is.null(generator$preprocessing_function)
  } else {
    FALSE
  }
}

is_main_thread_generator.keras_preprocessing.image.iterator.Iterator <- 
  is_main_thread_generator.keras_preprocessing.image.Iterator

is_tensorflow_dataset <- function(x) {
  inherits(x, "tensorflow.python.data.ops.dataset_ops.DatasetV2") ||
    inherits(x, "tensorflow.python.data.ops.dataset_ops.Dataset")
}

resolve_tensorflow_dataset <- function(x) {
  
  if (is_tensorflow_dataset(x)) {
    
    # check version compatibility
    
    if (is_tensorflow_implementation()) {
      if (tensorflow::tf_version() < "1.9")
        stop("TensorFlow v1.9 or higher is required for direct tensor input to models", call. = FALSE)
    } else {
      if (keras_version() < "2.2.0")
        stop("Keras v2.2 or higher is required for direct tensor input to models", call. = FALSE)
      if (!is_backend("tensorflow"))
        stop("The tensorflow backend is required for direct tensor input to models", call. = FALSE)
      if (tensorflow::tf_version() < "1.8")
        stop("TensorFlow v1.8 or higher is required for direct tensor input to models", call. = FALSE)
    }
    
    
    if (tensorflow::tf_version() < "1.14.0") {
      # yield iterators
      iter = x$make_one_shot_iterator()
      iter$get_next()  
    } else {
      x
    }
    
  } else {
    NULL
  }
}



  
#' Retrieves a layer based on either its name (unique) or index.
#'
#' Indices are based on order of horizontal graph traversal (bottom-up) and are
#' 1-based. If `name` and `index` are both provided, `index` will take
#' precedence.
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
  
  # convert to layer index
  index <- as_layer_index(index)
  
  # check for 0
  if (identical(index, -1L))
    stop("Indexes for get_layer() are 1-based (0 was passed as the index)")
  
  # call get_layer
  object$get_layer(
    name = name,
    index = index
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
  !is.null(metrics)     &&            # have metrics
  (length(metrics) > 0) &&            # capturing at least one metric
  (verbose > 0) &&                    # verbose mode is on
  !is.null(getOption("viewer")) &&    # have an internal viewer available
  nzchar(Sys.getenv("RSTUDIO"))       # running under RStudio
}


write_history_metadata <- function(history) {
  properties <- list()
  properties$validation_samples <- history$params$validation_samples
  tfruns::write_run_metadata("properties", properties)
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



