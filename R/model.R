
#' Keras Model
#'
#' A model is a directed acyclic graph of layers.
#'
#' @param inputs Input layer
#' @param outputs Output layer
#' @param ... Any additional arguments
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
keras_model <- function(inputs, outputs = NULL, ...) {
  if (tf_version() < "2.4")
    names(inputs) <- names(outputs) <- NULL

  keras$models$Model(inputs = inputs, outputs = outputs, ...)
}


#' Keras Model composed of a linear stack of layers
#'
#' @param layers List of layers to add to the model
#' @param name Name of model
#' @inheritDotParams sequential_model_input_layer
#'
#' @note
#'
#' If any arguments are provided to `...`, then the sequential model is
#' initialized with a `InputLayer` instance. If not, then the first layer passed
#' to a Sequential model should have a defined input shape. What that means is
#' that it should have received an `input_shape` or `batch_input_shape`
#' argument, or for some type of layers (recurrent, Dense...) an `input_dim`
#' argument.
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
#'
#' # alternative way to provide input shape
#' model <- keras_model_sequential(input_shape = c(784)) %>%
#'   layer_dense(units = 32) %>%
#'   layer_activation('relu') %>%
#'   layer_dense(units = 10) %>%
#'   layer_activation('softmax')
#'
#' }
#' @export
keras_model_sequential <- function(layers = NULL, name = NULL, ...) {

  if (length(list(...)))
    layers <- c(sequential_model_input_layer(...), layers)

  keras$models$Sequential(layers = layers, name = name)
}




#' sequential_model_input_layer
#'
#' @param input_shape an integer vector of dimensions (not including the batch
#'   axis), or a `tf$TensorShape` instance (also not including the batch axis).
#' @param batch_size  Optional input batch size (integer or NULL).
#' @param dtype Optional datatype of the input. When not provided, the Keras
#'   default float type will be used.
#' @param input_tensor Optional tensor to use as layer input. If set, the layer
#'   will use the `tf$TypeSpec` of this tensor rather than creating a new
#'   placeholder tensor.
#' @param sparse Boolean, whether the placeholder created is meant to be sparse.
#'   Default to `FALSE`.
#' @param ragged Boolean, whether the placeholder created is meant to be ragged.
#'   In this case, values of 'NULL' in the 'shape' argument represent ragged
#'   dimensions. For more information about `RaggedTensors`, see this
#'   [guide](https://www.tensorflow.org/guide/ragged_tensor). Default to
#'   `FALSE`.
#' @param type_spec A `tf$TypeSpec` object to create Input from. This
#'   `tf$TypeSpec` represents the entire batch. When provided, all other args
#'   except name must be `NULL`.
#' @param ... additional arguments passed on to `keras$layers$InputLayer`.
#' @param input_layer_name,name  Optional name of the input layer (string).
#'
sequential_model_input_layer <- function(input_shape = NULL,
                                         batch_size = NULL,
                                         dtype = NULL,
                                         input_tensor = NULL,
                                         sparse = NULL,
                                         name = NULL,
                                         ragged = NULL,
                                         type_spec = NULL,
                                         ...,
                                         input_layer_name = NULL) {
  # keras$layers$Input can't be used with a Sequential Model, have to use
  # keras$layers$LayerInput instead.
  args <- capture_args(match.call(),
                       list(input_shape = as_shape,
                            batch_size = as_nullable_integer))

  if ("input_layer_name" %in% names(args)) {
    # a bare `name` arg would normally belong to the model, not the input layer
    if (!is.null(args[["input_layer_name"]]))
      args[["name"]] <- args[["input_layer_name"]]

    args[["input_layer_name"]] <- NULL
  }

  do.call(keras$layers$InputLayer, args)
}



#' (Deprecated) Replicates a model on different GPUs.
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
#' @note This function is deprecated and has been removed from tensorflow on
#' 2020-04-01. To distribute your training across all available GPUS,
#' you can use `tensorflow::tf$distribute$MirroredStrategy()`
#' by creating your model like this:
#' ```r
#' strategy <- tensorflow::tf$distribute$MirroredStrategy()
#' with(strategy$scope(), {
#'   model <- application_xception(
#'     weights = NULL,
#'     input_shape = c(height, width, 3),
#'     classes = num_classes
#' })
#' ```
#' @keywords internal
#' @export
multi_gpu_model <- function(model, gpus = NULL, cpu_merge = TRUE, cpu_relocation = FALSE) {

  if (is.null(gpus) && keras_version() < "2.1.4") {
    stop("You must provide an explicit gpus argument in Keras versions ",
         "prior to 2.1.4")
  }

  if (tensorflow::tf_version() >= "2.2")
    stop("This function is deprecated as of TF version 2.2")

  args <- list(
    model = model,
    gpus = as_nullable_integer(gpus)
  )

  if (keras_version() >= "2.1.6") {
    args$cpu_merge <- cpu_merge
    args$cpu_relocation <- cpu_relocation
  }

  do.call(resolve_utils()$multi_gpu_model, args)
}


#' @importFrom reticulate py_to_r_wrapper
#' @export
py_to_r_wrapper.keras.engine.training.Model <- function(x) {
  force(x)
  function(object, ...) {
    compose_layer(object, x, ...)
  }
}

#' @export
py_to_r_wrapper.kerastools.model.RModel <- function(x) {
  force(x)
  function(...) {
    x$call(...)
  }
}


#' @export
py_to_r_wrapper.keras.engine.base_layer.Layer <- function(x) {
  force(x)
  function(object, ...) {
    if(missing(object))
      x(...)
    else
      compose_layer(object, x, ...)
  }
}


#  py_to_r_wrapper.keras.engine.base_layer.Layer <- function(x) {
#    force(x)
#    function(...) {
#      if(!missing(..1) && inherits(..1, "keras.engine.sequential.Sequential")) {
#        if(length(list(...)) > 1)
#          warning("Other arguments to ... are ignored because layer instance already created")
#        model <- ..1
#        model$add(x)
#        model
#      } else
#        x(...)
#    }
#  }


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
#' @param clone_function Callable to be used to clone each layer in the target
#'   model (except `InputLayer` instances). It takes as argument the layer
#'   instance to be cloned, and returns the corresponding layer instance to be
#'   used in the model copy. If unspecified, this callable defaults to the
#'   following serialization/deserialization function:
#'
#'   ```function(layer) layer$`__class__`$from_config(layer$get_config())```
#'
#'   By passing a custom callable, you can customize your copy of the model,
#'   e.g. by wrapping certain layers of interest (you might want to replace all
#'   LSTM instances with equivalent `Bidirectional(LSTM(...))` instances, for
#'   example).
#'
#' @export
clone_model <- function(model, input_tensors = NULL, clone_function = NULL) {
  args <- capture_args(match.call())
  do.call(keras$models$clone_model, args)
}


#' Configure a Keras model for training
#'
#' @param object Model object to compile.
#' @param optimizer String (name of optimizer) or optimizer instance. For most
#'   models, this defaults to `"rmsprop"`
#' @param loss String (name of objective function), objective function or a
#'   `keras$losses$Loss` subclass instance. An objective function is any
#'   callable with the signature `loss = fn(y_true, y_pred)`, where y_true =
#'   ground truth values with shape = `[batch_size, d0, .. dN]`, except sparse
#'   loss functions such as sparse categorical crossentropy where shape =
#'   `[batch_size, d0, .. dN-1]`. y_pred = predicted values with shape =
#'   `[batch_size, d0, .. dN]`. It returns a weighted loss float tensor. If a
#'   custom `Loss` instance is used and reduction is set to `NULL`, return value
#'   has the shape `[batch_size, d0, .. dN-1]` i.e. per-sample or per-timestep
#'   loss values; otherwise, it is a scalar. If the model has multiple outputs,
#'   you can use a different loss on each output by passing a dictionary or a
#'   list of losses. The loss value that will be minimized by the model will
#'   then be the sum of all individual losses, unless `loss_weights` is
#'   specified.
#' @param metrics List of metrics to be evaluated by the model during training
#'   and testing. Each of this can be a string (name of a built-in function),
#'   function or a `keras$metrics$Metric` class instance. See
#'   `?tf$keras$metrics`. Typically you will use `metrics=list('accuracy')`. A
#'   function is any callable with the signature `result = fn(y_true, y_pred)`.
#'   To specify different metrics for different outputs of a multi-output model,
#'   you could also pass a dictionary, such as `metrics=list(output_a =
#'   'accuracy', output_b = c('accuracy', 'mse'))`. You can also pass a list to
#'   specify a metric or a list of metrics for each output, such as
#'   `metrics=list(list('accuracy'), list('accuracy', 'mse'))` or
#'   `metrics=list('accuracy', c('accuracy', 'mse'))`. When you pass the strings
#'   `'accuracy'` or `'acc'`, this is converted to one of
#'   `tf.keras.metrics.BinaryAccuracy`, `tf.keras.metrics.CategoricalAccuracy`,
#'   `tf.keras.metrics.SparseCategoricalAccuracy` based on the loss function
#'   used and the model output shape. A similar conversion is done for the
#'   strings `'crossentropy'` and `'ce'`.
#' @param loss_weights  Optional list, dictionary, or named vector specifying
#'   scalar numeric coefficients to weight the loss contributions of different
#'   model outputs. The loss value that will be minimized by the model will then
#'   be the *weighted sum* of all individual losses, weighted by the
#'   `loss_weights` coefficients. If a list, it is expected to have a 1:1
#'   mapping to the model's outputs. If a dict, it is expected to map output
#'   names (strings) to scalar coefficients.
#' @param weighted_metrics List of metrics to be evaluated and weighted by
#'   `sample_weight` or `class_weight` during training and testing.
#' @param run_eagerly Bool. Defaults to `FALSE`. If `TRUE`, this Model's logic
#'   will not be wrapped in a `tf.function`. Recommended to leave this as `NULL`
#'   unless your Model cannot be run inside a `tf.function`. `run_eagerly=True`
#'   is not supported when using
#'   `tf.distribute.experimental.ParameterServerStrategy`. If the model's logic
#'   uses tensors in R control flow expressions like `if` and `for`, the model
#'   is still traceable with `tf.function`, but you will have to enter a
#'   `tfautograph::autograph({})` directly.
#' @param steps_per_execution Int. Defaults to 1. The number of batches to run
#'   during each `tf.function` call. Running multiple batches inside a single
#'   `tf.function` call can greatly improve performance on TPUs or small models
#'   with a large Python/R overhead. At most, one full epoch will be run each
#'   execution. If a number larger than the size of the epoch is passed, the
#'   execution will be truncated to the size of the epoch. Note that if
#'   `steps_per_execution` is set to `N`, `Callback.on_batch_begin` and
#'   `Callback.on_batch_end` methods will only be called every `N` batches (i.e.
#'   before/after each `tf.function` execution).
#' @param ... Arguments supported for backwards compatibility only.
#' @param sample_weight_mode If you need to do timestep-wise sample weighting
#'   (2D weights), set this to "temporal". `NULL` defaults to sample-wise
#'   weights (1D). If the model has multiple outputs, you can use a different
#'   `sample_weight_mode` on each output by passing a list of modes.
#' @param target_tensors By default, Keras will create a placeholder for the
#'   model's target, which will be fed with the target data during training. If
#'   instead you would like to use your own target tensor (in turn, Keras will
#'   not expect external data for these targets at training time), you can
#'   specify them via the `target_tensors` argument. It should be a single
#'   tensor (for a single-output sequential model).
#'
#' @family model functions
#'
#' @export
compile.keras.engine.training.Model <-
  function(object,
           optimizer = NULL,
           loss = NULL,
           metrics = NULL,
           loss_weights = NULL,
           weighted_metrics = NULL,
           run_eagerly = NULL,
           steps_per_execution = NULL,
           ...,
           target_tensors = NULL,
           sample_weight_mode = NULL) {

    # give losses a name
    loss_name <- substitute(loss)
    if (is.function(loss) &&
        !inherits(loss, "python.builtin.object") &&
        is.null(attr(loss, "py_function_name", TRUE)))
      attr(loss, "py_function_name") <- as_py_name(loss_name)

    # handle metrics
    if (!is.null(metrics)) {
      if(inherits(metrics, "python.builtin.object") ||
         is.function(metrics))
        metrics <- list(metrics)
      # convert metrics to list if it isn't one
      if(is.character(metrics))
        metrics <- as.list(metrics)

      # get metric names (if any)
      metric_names <- names(metrics)
      if (is.null(metric_names))
        metric_names <- rep_len("", length(metrics))

      # if all the metrics names are output names then leave them alone
      # (just convert to a list with no special processing)
      if (py_has_attr(object, "output_names") &&
          all(metric_names %in% object$output_names)) {
        metrics <- as.list(metrics)
      } else {
        # convert metrics to a list (adding names to any custom functions)
        metrics <- lapply(1:length(metrics), function(i) {
          metric <- metrics[[i]]

          if (is.function(metric) && nzchar(metric_names[[i]])) {
            warning(
              "Passing names for custom metrics is deprecated. Please use the ",
              "custom_metric() function to define custom metrics."
            )
            attr(metric, "py_function_name") <- metric_names[[i]]
          }

          metric
        })
      }
    }

    # keras 2.07 args
    if (keras_version() >= "2.0.7") {
      # weighted metrics
      if (!is.null(weighted_metrics) && !is.list(weighted_metrics))
        weighted_metrics <- list(weighted_metrics)

      # target tensors
      if (!is.null(target_tensors) && !is.list(target_tensors))
        target_tensors <- list(target_tensors)
    }

    if (is.numeric(loss_weights))
      storage.mode(loss_weights) <- "list"

    args <- list(
      optimizer = optimizer,
      loss = loss,
      metrics = metrics,
      loss_weights = loss_weights,
      weighted_metrics = weighted_metrics,
      run_eagerly = run_eagerly,
      steps_per_execution = steps_per_execution,
      sample_weight_mode = sample_weight_mode,
      target_tensors = target_tensors
    )

    # drop NULLs
    for (nm in names(args))
      args[[nm]] <- args[[nm]]

    args <- c(list(), args, ...)

    # compile model
    do.call(object$compile, args)

    # return model invisible (convenience for chaining)
    invisible(object)
  }

as_py_name <- function(x) {
  if(is.language(x))
    x <- deparse(x, width.cutoff = 500L)[1]
  x <- make.names(as.character(x))
  x <- gsub(".", "_", x, fixed = TRUE)
  x
}

#drop_nulls <-
function(x, ...) {
  nms <- c(...)
  nms <- if (length(nms))
    intersect(names(x), nms)
  else
    names(args)

  for (nm in nms)
    x[[nm]] <- x[[nm]]
  x
}


resolve_input_data <- function(x, y = NULL) {
  # resolve x and y (check for TF dataset)
  dataset <- resolve_tensorflow_dataset(x)
  args <- list()
  if (inherits(dataset, "tensorflow.python.data.ops.dataset_ops.DatasetV2")) {
    args$x <- dataset
  } else if (!is.null(dataset)) {
    args$x <- dataset[[1]]
    args$y <- dataset[[2]]
  } else if (is.function(x)) {
    args$x <- as_generator(x)
  } else if (inherits(x, "python.builtin.iterator")) {
    args$x <- x
  } else if (inherits(x, "keras.utils.data_utils.Sequence")) {
    args$x <- x
  } else {
    if (!is.null(x))
      args$x <- keras_array(x)
    if (!is.null(y))
      args$y <- keras_array(y)
  }
  args
}

resolve_validation_data <- function(validation_data) {
  args <- list()
  if (!is.null(validation_data)) {
    dataset <- resolve_tensorflow_dataset(validation_data)
    if (!is.null(dataset))
      args$validation_data <- dataset
    else if (is.function(validation_data))
      args$validation_data <- as_generator(validation_data)
    else if (inherits(validation_data, "python.builtin.iterator"))
      args$validation_data <- validation_data
    else if (inherits(validation_data, "keras.utils.data_utils.Sequence"))
      args$validation_data <- validation_data
    else {
      args$validation_data <- keras_array(validation_data)
      if (tensorflow::tf_version() >="2.2")
        args$validation_data <- do.call(reticulate::tuple, args$validation_data)
    }
  }
  args
}

resolve_main_thread_generators <- function(x, callback_type = "on_train_batch_begin") {

  if (tensorflow::tf_version() == "2.1")
    stop("Using generators that call R functions is not supported in TensorFlow 2.1 ",
         "Please upgrade your TF installation or downgrade to 2.0", call. = FALSE)

  # we need a hack to make sure the generator is evaluated in the main thread.
  python_path <- system.file("python", package = "keras")
  tools <- reticulate::import_from_path("kerastools", path = python_path)

  # as_generator will return a tuple with 2 elements.
  # (1) a python generator that just consumes
  # a queue.
  # (2) a function that evaluates the next element of the generator
  # and adds to the queue. This function should be called in the main
  # thread.
  # we add a `on_train_batch_begin` to call this function.
  o <- tools$model$as_generator(x)

  callback <- list(function(batch, logs) {
    o[[2]]()
  })
  names(callback) <- callback_type

  if (callback_type == "on_test_batch_begin") {
    callback[[2]] <- callback[[1]]
    names(callback)[[2]] <- "on_test_begin"
  }

  callback <- do.call(callback_lambda, callback)

  list(generator = o[[1]], callback = callback)
}

#' Train a Keras model
#'
#' Trains the model for a fixed number of epochs (iterations on a dataset).
#'
#' @param object Model to train.
#' @param x Vector, matrix, or array of training data (or list if the model has
#'   multiple inputs). If all inputs in the model are named, you can also pass a
#'   list mapping input names to data. `x` can be `NULL` (default) if feeding
#'   from framework-native tensors (e.g. TensorFlow data tensors). You can also
#'   pass a `tfdataset` or a generator returning a list with `(inputs, targets)` or
#'   `(inputs, targets, sample_weights)`.
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

  if (!is.null(batch_size) && is_tensorflow_dataset(x))
    stop("Don't set batch_size with a tfdataset as input.", call. = FALSE)

  # defaults
  if (is.null(batch_size) && is.null(steps_per_epoch) && !is_tensorflow_dataset(x))
    batch_size <- 32L

  # resolve view_metrics
  if (identical(view_metrics, "auto"))
    view_metrics <- resolve_view_metrics(verbose, epochs, object$metrics)

  # build args
  args <- list(
    batch_size = as_nullable_integer(batch_size),
    epochs = as.integer(epochs),
    verbose = as.integer(verbose),
    validation_split = validation_split,
    shuffle = shuffle,
    class_weight = as_class_weight(class_weight),
    sample_weight = keras_array(sample_weight),
    initial_epoch = as.integer(initial_epoch)
  )

  args <- append(args, resolve_input_data(x, y))
  args <- append(args, resolve_validation_data(validation_data))

  if (keras_version() >= "2.0.7") {
    args$steps_per_epoch <- as_nullable_integer(steps_per_epoch)
    args$validation_steps <- as_nullable_integer(validation_steps)
  }

  extra_callbacks <- list()
  if (is_main_thread_generator(x)) {
    main_thr <- resolve_main_thread_generators(args$x)
    args$x <- main_thr$generator
    extra_callbacks <- c(extra_callbacks, main_thr$callback)
  }

  if (is_main_thread_generator(validation_data)) {
    main_thr <- resolve_main_thread_generators(args$validation_data, "on_test_batch_begin")
    args$validation_data <- main_thr$generator
    extra_callbacks <- c(extra_callbacks, main_thr$callback)
  }

  if (length(extra_callbacks) > 0) {
    callbacks <- c(callbacks, extra_callbacks)
  }

  args$callbacks <- normalize_callbacks_with_metrics(view_metrics, initial_epoch, callbacks)
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
#'   from framework-native tensors (e.g. TensorFlow data tensors). You can also
#'   pass a `tfdataset` or a generator returning a list with `(inputs, targets)` or
#'   `(inputs, targets, sample_weights)`.
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

  args <- append(args, resolve_input_data(x, y))

  extra_callbacks <- list()
  if (is_main_thread_generator(x)) {
    main_thr <- resolve_main_thread_generators(args$x, "on_test_batch_begin")
    args$x <- main_thr$generator
    extra_callbacks <- c(extra_callbacks, main_thr$callback)
  }

  if (length(extra_callbacks) > 0) {
    callbacks <- c(callbacks, extra_callbacks)
  }

  args <- resolve_callbacks(args, callbacks)

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
#' @param x Input data (vector, matrix, or array). You can also
#'   pass a `tfdataset` or a generator returning a list with `(inputs, targets)` or
#'   `(inputs, targets, sample_weights)`.
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

  args <- append(args, resolve_input_data(x))

  extra_callbacks <- list()
  if (is_main_thread_generator(x)) {
    main_thr <- resolve_main_thread_generators(args$x, "on_predict_batch_begin")
    args$x <- main_thr$generator
    extra_callbacks <- c(extra_callbacks, main_thr$callback)
  }

  if (length(extra_callbacks) > 0) {
    callbacks <- c(callbacks, extra_callbacks)
  }

  args <- resolve_callbacks(args, callbacks)

  if (keras_version() >= "2.0.7")
    args$steps <- as_nullable_integer(steps)

  # call predict
  do.call(object$predict, args)
}


#' (Deprecated) Generates probability or class probability predictions for the input samples.
#'
#' These functions were removed in Tensorflow version 2.6. See details for how to update your code:
#'
#' @details How to update your code:
#'
#' `predict_proba()`: use `predict()` directly.
#'
#' `predict_classes()`:
#'   * If your model does multi-class classification:
#'     (e.g. if it uses a `softmax` last-layer activation).
#'  ```r
#'       model %>% predict(x) %>% k_argmax()
#'  ```
#'   * if your model does binary classification
#'     (e.g. if it uses a `sigmoid` last-layer activation).
#'  ```r
#'       model %>% predict(x) %>% `>`(0.5) %>% k_cast("int32")
#'  ```
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
#' @keywords internal
#' @export
predict_proba <- function(object, x, batch_size = NULL, verbose = 0, steps = NULL) {
  warning("`predict_proba()` is deprecated and was removed from tensorflow in version 2.6, ",
          "please use `predict()` instead")
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
#' @keywords internal
#' @export
predict_classes <- function(object, x, batch_size = NULL, verbose = 0, steps = NULL) {
  warning(
'`predict_classes()` is deprecated and and was removed from tensorflow in version 2.6.
Please update your code:
  * If your model does multi-class classification:
    (e.g. if it uses a `softmax` last-layer activation).

      model %>% predict(x) %>% k_argmax()

  * if your model does binary classification
    (e.g. if it uses a `sigmoid` last-layer activation).

      model %>% predict(x) %>% `>`(0.5) %>% k_cast("int32")
'
  )
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



#' (Deprecated) Fits the model on data yielded batch-by-batch by a generator.
#'
#' The generator is run in parallel to the model, for efficiency. For instance,
#' this allows you to do real-time data augmentation on images on CPU in
#' parallel to training your model on GPU.
#'
#' @inheritParams fit.keras.engine.training.Model
#'
#' @param object Keras model object
#' @param generator A generator (e.g. like the one provided by
#'   [flow_images_from_directory()] or a custom R
#'   [generator function](https://rstudio.github.io/reticulate/articles/calling_python.html#generators-1)).
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
#' @keywords internal
#' @export
fit_generator <- function(object, generator, steps_per_epoch, epochs = 1,
                          verbose=getOption("keras.fit_verbose", default = 1), callbacks = NULL,
                          view_metrics = getOption("keras.view_metrics", default = "auto"),
                          validation_data = NULL, validation_steps = NULL,
                          class_weight = NULL, max_queue_size = 10, workers = 1, initial_epoch = 0) {

  if (tensorflow::tf_version() <= "2.0")
    return(fit_generator_legacy(
      object = object,
      generator = generator,
      steps_per_epoch = steps_per_epoch,
      epochs = epochs,
      verbose=verbose,
      view_metrics = view_metrics,
      validation_data = validation_data,
      validation_steps = validation_steps,
      class_weight = class_weight,
      max_queue_size = max_queue_size,
      workers = workers,
      initial_epoch = initial_epoch
    ))

  warning("`fit_generator` is deprecated. Use `fit` instead, it now accept generators.")

  # redirect to `model.fit`
  args <- list(
    object = object,
    x = generator,
    steps_per_epoch = steps_per_epoch,
    epochs = epochs,
    verbose = verbose,
    callbacks = callbacks,
    validation_data = validation_data,
    validation_steps = validation_steps,
    class_weight = class_weight,
    max_queue_size = max_queue_size,
    workers = workers,
    initial_epoch = initial_epoch
  )

  do.call(fit, args)
}

#' (Deprecated) Evaluates the model on a data generator.
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
#' @keywords internal
#' @export
evaluate_generator <- function(object, generator, steps, max_queue_size = 10, workers = 1,
                               callbacks = NULL) {

  if (tensorflow::tf_version() <= "2.0")
    return(evaluate_generator_legacy(
      object, generator, steps, max_queue_size, workers,
      callbacks))

  warning("`evaluate_generator` is deprecated. Use `evaluate` instead, it now accept generators.")

  args <- list(
    object = object,
    x = generator,
    steps = as.integer(steps),
    max_queue_size = as.integer(max_queue_size),
    workers = as.integer(workers),
    callbacks = callbacks
  )

  do.call(evaluate, args)
}


#' (Deprecated) Generates predictions for the input samples from a data generator.
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
#' @keywords internal
#' @export
predict_generator <- function(object, generator, steps, max_queue_size = 10, workers = 1, verbose = 0,
                              callbacks = NULL) {

  if (tensorflow::tf_version() <= "2.0")
    return(predict_generator_legacy(object, generator, steps, max_queue_size,
                             workers, verbose, callbacks))

  warning("`predict_generator` is deprecated. Use `predict` instead, it now accept generators.")

  args <- list(
    object = object,
    x = generator,
    steps = as.integer(steps),
    max_queue_size = as.integer(max_queue_size),
    workers = as.integer(workers),
    verbose = as.integer(verbose),
    callbacks = callbacks
  )

  do.call(predict, args)
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

  if (tensorflow::tf_version() <= "2.0.1")
    return(TRUE)

  if (py_has_attr(x, "image_data_generator")) {
    generator <- x$image_data_generator
    !is.null(generator$preprocessing_function)
  } else {
    FALSE
  }
}

is_main_thread_generator.keras_preprocessing.image.iterator.Iterator <-
  is_main_thread_generator.keras_preprocessing.image.Iterator

is_main_thread_generator.keras_preprocessing.sequence.TimeseriesGenerator <- function(x) {
  if (tensorflow::tf_version() <= "2.0.1")
    return(TRUE)

  FALSE
}

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
#' @param index Integer, index of layer (1-based)
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
#' @param object,x Keras model instance
#' @param line_length Total length of printed lines
#' @param positions Relative or absolute positions of log elements in each line.
#'   If not provided, defaults to `c(0.33, 0.55, 0.67, 1.0)`.
#' @param expand_nested Whether to expand the nested models. If not provided,
#'   defaults to `FALSE`.
#' @param show_trainable Whether to show if a layer is trainable. If not
#'   provided, defaults to `FALSE`.
#' @param ... for `summary()` and `print()`, passed on to `format()`. For
#'   `format()`, passed on to `model$summary()`.
#'
#' @family model functions
#'
#' @return `format()` returns a length 1 character vector. `print()` returns the
#'   model object invisibly. `summary()` returns the output of `format()`
#'   invisibly after printing it.
#'
#' @export
summary.keras.engine.training.Model <- function(object, ...) {
  writeLines(f <- format.keras.engine.training.Model(object, ...))
  invisible(f)
}

#' @rdname summary.keras.engine.training.Model
#' @export
format.keras.engine.training.Model <-
  function(x,
           line_length = getOption("width"),
           positions = NULL,
           expand_nested = FALSE,
           show_trainable = FALSE,
           ...) {
    if (py_is_null_xptr(x))
      return("<pointer: 0x0>")

    args <- capture_args(match.call(), ignore = "x")

    # ensure `line_length` in args, even if not passed by user
    args$line_length <- as_nullable_integer(line_length)

    if (x$built)
      trimws(py_capture_output(do.call(x$summary, args),
                               type = "stdout"))
     else
      "Model: <no summary available, model was not built>"
}

#
#' @rdname summary.keras.engine.training.Model
#' @export
print.keras.engine.training.Model <- function(x, ...) {
  writeLines(format.keras.engine.training.Model(x, ...))
  invisible(x)
}

#' @importFrom reticulate py_str
#' @export
py_str.keras.engine.training.Model <- function(object, line_length = getOption("width"), positions = NULL, ...) {
  # still invoked by utils::str()
  # warning("`py_str()` generic is deprecated")
  format.keras.engine.training.Model(object, line_length = line_length, positions = positions, ...)
}


# determine whether to view metrics or not
resolve_view_metrics <- function(verbose, epochs, metrics) {
  (epochs > 1)          &&            # more than 1 epoch
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
