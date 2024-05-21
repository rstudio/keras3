
# ---- compile ----
#' Configure a model for training.
#'
#' @description
#'
#' # Examples
#' ```r
#' model |> compile(
#'   optimizer = optimizer_adam(learning_rate = 1e-3),
#'   loss = loss_binary_crossentropy(),
#'   metrics = c(metric_binary_accuracy(),
#'               metric_false_negatives())
#' )
#' ```
#'
#' @param object A Keras model.
#'
#' @param optimizer
#' String (name of optimizer) or optimizer instance. See
#' `optimizer_*` family.
#'
#' @param loss
#' Loss function. May be:
#' - a string (name of builtin loss function),
#' - a custom function, or
#' - a [`Loss`] instance (returned by the `loss_*` family of functions).
#'
#' A loss function is any callable with the signature
#' `loss = fn(y_true, y_pred)`, where `y_true` are the ground truth
#' values, and `y_pred` are the model's predictions.
#' `y_true` should have shape `(batch_size, d1, .. dN)`
#' (except in the case of sparse loss functions such as
#' sparse categorical crossentropy which expects integer arrays of
#' shape `(batch_size, d1, .. dN-1)`).
#' `y_pred` should have shape `(batch_size, d1, .. dN)`.
#' The loss function should return a float tensor.
#'
#' @param loss_weights
#' Optional list (named or unnamed) specifying scalar
#' coefficients (R numerics) to weight the loss contributions of
#' different model outputs. The loss value that will be minimized
#' by the model will then be the *weighted sum* of all individual
#' losses, weighted by the `loss_weights` coefficients.  If an unnamed list,
#' it is expected to have a 1:1 mapping to the model's outputs. If
#' a named list, it is expected to map output names (strings) to scalar
#' coefficients.
#'
#' @param metrics
#' List of metrics to be evaluated by the model during
#' training and testing. Each of these can be:
#' - a string (name of a
#' built-in function),
#' - a function, optionally with a `"name"` attribute or
#' - a [`Metric()`]
#' instance. See the `metric_*` family of functions.
#'
#' Typically you will use
#' `metrics = c('accuracy')`. A function is any callable with the
#' signature `result = fn(y_true, y_pred)`. To specify different
#' metrics for different outputs of a multi-output model, you could
#' also pass a named list, such as
#' `metrics = list(a = 'accuracy', b = c('accuracy', 'mse'))`.
#' You can also pass a list to specify a metric or a list of
#' metrics for each output, such as
#' `metrics = list(c('accuracy'), c('accuracy', 'mse'))`
#' or `metrics = list('accuracy', c('accuracy', 'mse'))`. When you pass
#' the strings `'accuracy'` or `'acc'`, we convert this to one of
#' `metric_binary_accuracy()`,
#' `metric_categorical_accuracy()`,
#' `metric_sparse_categorical_accuracy()` based on the
#' shapes of the targets and of the model output. A similar
#' conversion is done for the strings `"crossentropy"`
#' and `"ce"` as well.
#' The metrics passed here are evaluated without sample weighting;
#' if you would like sample weighting to apply, you can specify
#' your metrics via the `weighted_metrics` argument instead.
#'
#' If providing an anonymous R function, you can customize the printed name
#' during training by assigning `attr(<fn>, "name") <- "my_custom_metric_name"`,
#' or by calling [`custom_metric("my_custom_metric_name", <fn>)`][`custom_metric()`]
#'
#' @param weighted_metrics
#' List of metrics to be evaluated and weighted by
#' `sample_weight` or `class_weight` during training and testing.
#'
#' @param run_eagerly
#' Bool. If `TRUE`, this model's forward pass
#' will never be compiled. It is recommended to leave this
#' as `FALSE` when training (for best performance),
#' and to set it to `TRUE` when debugging.
#'
#' @param steps_per_execution
#' Int. The number of batches to run
#' during each a single compiled function call. Running multiple
#' batches inside a single compiled function call can
#' greatly improve performance on TPUs or small models with a large
#' R/Python overhead. At most, one full epoch will be run each
#' execution. If a number larger than the size of the epoch is
#' passed, the execution will be truncated to the size of the
#' epoch. Note that if `steps_per_execution` is set to `N`,
#' `Callback$on_batch_begin` and `Callback$on_batch_end` methods
#' will only be called every `N` batches (i.e. before/after
#' each compiled function execution).
#' Not supported with the PyTorch backend.
#'
#' @param jit_compile
#' Bool or `"auto"`. Whether to use XLA compilation when
#' compiling a model. For `jax` and `tensorflow` backends,
#' `jit_compile="auto"` enables XLA compilation if the model
#' supports it, and disabled otherwise.
#' For `torch` backend, `"auto"` will default to eager
#' execution and `jit_compile=True` will run with `torch.compile`
#' with the `"inductor"` backend.
#'
#' @param auto_scale_loss
#' Bool. If `TRUE` and the model dtype policy is
#' `"mixed_float16"`, the passed optimizer will be automatically
#' wrapped in a `LossScaleOptimizer`, which will dynamically
#' scale the loss to prevent underflow.
#'
#' @returns This is called primarily for the side effect of modifying `object`
#'   in-place. The first argument `object` is also returned, invisibly, to
#'   enable usage with the pipe.
#'
#' @param object Keras model object
#' @param ... Additional arguments passed on to the `compile()` model method.
#' @export
#' @tether keras.Model.compile
#' @family model training
#' @seealso
#' + <https://keras.io/api/models/model_training_apis#compile-method>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/Model/compile>
compile.keras.src.models.model.Model <-
function (object, optimizer = "rmsprop", loss = NULL, metrics = NULL,
          ..., loss_weights = NULL, weighted_metrics = NULL,
          run_eagerly = FALSE,
          steps_per_execution = 1L,
          jit_compile = "auto",
          auto_scale_loss = TRUE)
{
  args <- capture_args(list(
    steps_per_execution = as_integer,
    loss = as_loss,
    metrics = as_metrics,
    weighted_metrics = as_list,
    loss_weights = as.list
  ),
  ignore = "object")

  do.call(object$compile, args)

  # return model invisible (convenience for chaining)
  invisible(object)
}

as_loss <- function(x, default_name = "custom_loss") {
  if (is.null(x) || is_string(x))
    return(x)
  if (is.character(x)) # failed is_string(x), length(x) != 1
    return(as.list(x))
  if (is.list(x)) # recurse for multi-output models
    return(imap(x, function(el, i) {
      as_loss(el, default_name = paste(default_name, i, sep = "_"))
    }))
  resolve_py_obj(x, default_name = default_name, prefer_class = FALSE)
}

as_metrics <- function(x) as_list(as_loss(x, default_name = "custom_metric"))


# ---- evaluate ----


#' Evaluate a Keras Model
#'
#' @description
#' This functions returns the loss value and metrics values for the model in
#' test mode.
#' Computation is done in batches (see the `batch_size` arg.)
#'
#' @returns
#' Scalar test loss (if the model has a single output and no metrics)
#' or list of scalars (if the model has multiple outputs
#' and/or metrics). The attribute `model$metrics_names` will give you
#' the display labels for the scalar outputs.
#'
#' @param x
#' Input data. It could be:
#' - An R array (or array-like), or a list of arrays
#'     (in case the model has multiple inputs).
#' - A tensor, or a list of tensors
#'     (in case the model has multiple inputs).
#' - A named list mapping input names to the corresponding array/tensors,
#'     if the model has named inputs.
#' - A `tf.data.Dataset`. Should return a tuple
#'     of either `(inputs, targets)` or
#'     `(inputs, targets, sample_weights)`.
#' - A generator returning
#'     `(inputs, targets)` or `(inputs, targets, sample_weights)`.
#'
#' @param y
#' Target data. Like the input data `x`, it could be either R
#' array(s) or backend-native tensor(s).
#' If `x` is a `tf.data.Dataset` or generator function,
#' `y` should not be specified
#' (since targets will be obtained from the iterator/dataset).
#'
#' @param batch_size
#' Integer or `NULL`. Number of samples per batch of
#' computation. If unspecified, `batch_size` will default to `32`. Do
#' not specify the `batch_size` if your data is in the form of a
#' a tf dataset or generator
#' (since they generate batches).
#'
#' @param verbose
#' `"auto"`, `0`, `1`, or `2`. Verbosity mode.
#' `0` = silent, `1` = progress bar, `2` = single line.
#' `"auto"` becomes `1` for most cases,
#' `2` if in a knitr render or running on a distributed training server.
#' Note that the progress bar is not
#' particularly useful when logged to a file, so `verbose=2` is
#' recommended when not running interactively
#' (e.g. in a production environment). Defaults to `"auto"`.
#'
#' @param sample_weight
#' Optional array of weights for the test samples,
#' used for weighting the loss function. You can either pass a flat
#' (1D) R array with the same length as the input samples
#' (1:1 mapping between weights and samples), or in the case of
#' temporal data, you can pass a 2D array with shape `(samples,
#' sequence_length)`, to apply a different weight to every
#' timestep of every sample. This argument is not supported when
#' `x` is a tfdataset, instead pass sample weights as the third
#' element of `x`.
#'
#' @param steps
#' Integer or `NULL`. Total number of steps (batches of samples)
#' before declaring the evaluation round finished. Ignored with the
#' default value of `NULL`. If `x` is a `tf.data.Dataset` and
#' `steps` is `NULL`, evaluation will run until the dataset
#' is exhausted.
#'
#' @param callbacks
#' List of `Callback` instances.
#' List of callbacks to apply during evaluation.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @param object Keras model object
#'
#' @export
#' @tether keras.Model.evaluate
#' @family model training
#' @seealso
#' + <https://keras.io/api/models/model_training_apis#evaluate-method>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/Model/evaluate>
evaluate.keras.src.models.model.Model <-
function (object, x = NULL, y = NULL, ..., batch_size = NULL,
          verbose = getOption("keras.verbose", default = "auto"),
          sample_weight = NULL, steps = NULL, callbacks = NULL)
{
    normalize_input_data <- input_data_normalizer(object)
    args <- capture_args(list(x = normalize_input_data,
                               y = normalize_input_data,
                               sample_weight = normalize_input_data,
                               batch_size = as_integer, steps = as_integer,
                               verbose = as_model_verbose_arg),
                          ignore = "object",
                          force = "verbose")

    ## return_dict=TRUE because object$metrics_names returns wrong value
    ## (e.g., "compile_metrics" instead of "mae")
    args[["return_dict"]] <- TRUE

    if(inherits(args$x, "tensorflow.python.data.ops.dataset_ops.DatasetV2") &&
       !is.null(args$batch_size))
      stop("batch_size can not be specified with a TF Dataset")

    result <- do.call(object$evaluate, args)
    # if (length(result) > 1L) { ## if return_dict=FALSE
    #   result <- as.list(result)
    #   names(result) <- object$metrics_names
    # }

    tfruns::write_run_metadata("evaluation", unlist(result))

    result
}


# ---- fit ----
#' Train a model for a fixed number of epochs (dataset iterations).
#'
#' @details
#' Unpacking behavior for iterator-like inputs:
#'
#' A common pattern is to pass an iterator like object such as a
#' `tf.data.Dataset` or a generator to `fit()`,
#' which will in fact yield not only features (`x`)
#' but optionally targets (`y`) and sample weights (`sample_weight`).
#' Keras requires that the output of such iterator-likes be
#' unambiguous. The iterator should return a `tuple()`
#' of length 1, 2, or 3, where the optional second and third elements
#' will be used for `y` and `sample_weight` respectively.
#' Any other type provided will be wrapped in
#' a length-one `tuple()`, effectively treating everything as `x`. When
#' yielding named lists, they should still adhere to the top-level tuple
#' structure,
#' e.g. `tuple(list(x0 = x0, x = x1), y)`. Keras will not attempt to separate
#' features, targets, and weights from the keys of a single dict.
#'
#' @returns
#' A `keras_training_history` object, which is a named list:
#' `list(params = <params>, metrics = <metrics>")`, with S3 methods
#' `print()`, `plot()`, and `as.data.frame()`. The metrics
#' field is
#' a record of training loss values and metrics values
#' at successive epochs, as well as validation loss values
#' and validation metrics values (if applicable).
#'
#' @param x
#' Input data. It could be:
#' - An array (or array-like), or a list of arrays
#'   (in case the model has multiple inputs).
#' - A tensor, or a list of tensors
#'   (in case the model has multiple inputs).
#' - A named list mapping input names to the corresponding array/tensors,
#'   if the model has named inputs.
#' - A `tf.data.Dataset`. Should return a tuple
#'   of either `(inputs, targets)` or
#'   `(inputs, targets, sample_weights)`.
#' - A generator returning `(inputs,
#'   targets)` or `(inputs, targets, sample_weights)`.
#'
#' @param y
#' Target data. Like the input data `x`,
#' it could be either array(s) or backend-native tensor(s).
#' If `x` is a TF Dataset or generator,
#' `y` should
#' not be specified (since targets will be obtained from `x`).
#'
#' @param batch_size
#' Integer or `NULL`.
#' Number of samples per gradient update.
#' If unspecified, `batch_size` will default to `32`.
#' Do not specify the `batch_size` if your data is in the
#' form of TF Datasets or generators,
#' (since they generate batches).
#'
#' @param epochs
#' Integer. Number of epochs to train the model.
#' An epoch is an iteration over the entire `x` and `y`
#' data provided
#' (unless the `steps_per_epoch` flag is set to
#' something other than `NULL`).
#' Note that in conjunction with `initial_epoch`,
#' `epochs` is to be understood as "final epoch".
#' The model is not trained for a number of iterations
#' given by `epochs`, but merely until the epoch
#' of index `epochs` is reached.
#'
#' @param verbose
#' `"auto"`, `0`, `1`, or `2`. Verbosity mode.
#' `0` = silent, `1` = progress bar, `2` = one line per epoch.
#' `"auto"` becomes 1 for most cases,
#' `2` if in a knitr render or running on a distributed training server.
#' Note that the progress bar is not
#' particularly useful when logged to a file,
#' so `verbose=2` is recommended when not running interactively
#' (e.g., in a production environment). Defaults to `"auto"`.
#'
#' @param callbacks
#' List of `Callback()` instances.
#' List of callbacks to apply during training.
#' See `callback_*`.
#'
#' @param validation_split
#' Float between 0 and 1.
#' Fraction of the training data to be used as validation data.
#' The model will set apart this fraction of the training data,
#' will not train on it, and will evaluate
#' the loss and any model metrics
#' on this data at the end of each epoch.
#' The validation data is selected from the last samples
#' in the `x` and `y` data provided, before shuffling. This
#' argument is not supported when `x` is a TF Dataset or generator.
#' If both `validation_data` and `validation_split` are provided,
#' `validation_data` will override `validation_split`.
#'
#' @param validation_data
#' Data on which to evaluate
#' the loss and any model metrics at the end of each epoch.
#' The model will not be trained on this data. Thus, note the fact
#' that the validation loss of data provided using
#' `validation_split` or `validation_data` is not affected by
#' regularization layers like noise and dropout.
#' `validation_data` will override `validation_split`.
#' It could be:
#'   - A tuple `(x_val, y_val)` of arrays or tensors.
#'   - A tuple `(x_val, y_val, val_sample_weights)` of
#'     arrays.
#'   - A generator returning
#'   `(inputs, targets)` or `(inputs, targets, sample_weights)`.
#'
#' @param shuffle
#' Boolean, whether to shuffle the training data
#' before each epoch. This argument is
#' ignored when `x` is a generator or a TF Dataset.
#'
#' @param class_weight
#' Optional named list mapping class indices (integers, 0-based)
#' to a weight (float) value, used for weighting the loss function
#' (during training only).
#' This can be useful to tell the model to
#' "pay more attention" to samples from
#' an under-represented class. When `class_weight` is specified
#' and targets have a rank of 2 or greater, either `y` must be
#' one-hot encoded, or an explicit final dimension of `1` must
#' be included for sparse class labels.
#'
# @param class_names
#'
#' @param sample_weight
#' Optional array of weights for
#' the training samples, used for weighting the loss function
#' (during training only). You can either pass a flat (1D)
#' array/vector with the same length as the input samples
#' (1:1 mapping between weights and samples),
#' or in the case of temporal data,
#' you can pass a 2D array (matrix) with shape
#' `(samples, sequence_length)`,
#' to apply a different weight to every timestep of every sample.
#' This argument is not supported when `x` is a TF Dataset or generator,
#' instead provide the
#' sample_weights as the third element of `x`.
#' Note that sample weighting does not apply to metrics specified
#' via the `metrics` argument in `compile()`. To apply sample
#' weighting to your metrics, you can specify them via the
#' `weighted_metrics` in `compile()` instead.
#'
#' @param initial_epoch
#' Integer.
#' Epoch at which to start training
#' (useful for resuming a previous training run).
#'
#' @param steps_per_epoch
#' Integer or `NULL`.
#' Total number of steps (batches of samples)
#' before declaring one epoch finished and starting the
#' next epoch. When training with input tensors such as
#' backend-native tensors, the default `NULL` is equal to
#' the number of samples in your dataset divided by
#' the batch size, or `1` if that cannot be determined. If `x` is a
#' TF Dataset, and `steps_per_epoch`
#' is `NULL`, the epoch will run until the input dataset is
#' exhausted.  When passing an infinitely repeating dataset, you
#' must specify the `steps_per_epoch` argument. If
#' `steps_per_epoch = -1` the training will run indefinitely with an
#' infinitely repeating dataset.
#'
#' @param validation_steps
#' Only relevant if `validation_data` is provided.
#' Total number of steps (batches of
#' samples) to draw before stopping when performing validation
#' at the end of every epoch. If `validation_steps` is `NULL`,
#' validation will run until the `validation_data` dataset is
#' exhausted. In the case of an infinitely repeated dataset, it
#' will run into an infinite loop. If `validation_steps` is
#' specified and only part of the dataset will be consumed, the
#' evaluation will start from the beginning of the dataset at each
#' epoch. This ensures that the same validation samples are used
#' every time.
#'
#' @param validation_batch_size
#' Integer or `NULL`.
#' Number of samples per validation batch.
#' If unspecified, will default to `batch_size`.
#' Do not specify the `validation_batch_size` if your data is in
#' the form of TF Datasets or generator
#' instances (since they generate batches).
#'
#' @param validation_freq
#' Only relevant if validation data is provided.
#' Specifies how many training epochs to run
#' before a new validation run is performed,
#' e.g. `validation_freq=2` runs validation every 2 epochs.
#'
#' @param object Keras model object
#'
#' @param view_metrics View realtime plot of training metrics (by epoch). The
#'   default (`"auto"`) will display the plot when running within RStudio,
#'   `metrics` were specified during model [compile()], `epochs > 1` and
#'   `verbose > 0`. Set the global `options(keras.view_metrics = )` option to
#'   establish a different default.
#'
#' @param ... Additional arguments passed on to the model `fit()` method.
#'
#' @export
#' @tether keras.Model.fit
#' @seealso
#' + <https://keras.io/api/models/model_training_apis#fit-method>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/Model/fit>
fit.keras.src.models.model.Model <-
function(object,
         x = NULL,
         y = NULL,
         ...,
         batch_size = NULL,
         epochs = 1L,
         callbacks = NULL,
         validation_split = 0,
         validation_data = NULL,
         shuffle = TRUE,
         class_weight = NULL,
         # class_names = names(class_weight),
         sample_weight = NULL,
         initial_epoch = 1L,
         steps_per_epoch = NULL,
         validation_steps = NULL,
         validation_batch_size = NULL,
         validation_freq = 1L,
         verbose = getOption("keras.verbose", default = "auto"),
         view_metrics = getOption("keras.view_metrics", default = "auto"))
{
  normalize_input_data <- input_data_normalizer(object)
    args <- capture_args(
        list(
            x = normalize_input_data,
            y =  normalize_input_data,
            sample_weight = normalize_input_data,
            validation_data = normalize_input_data,

            batch_size = as_integer,
            validation_batch_size = as_integer,
            epochs = as_integer,
            initial_epoch = as_index,
            steps_per_epoch = as_integer,
            validation_freq = as_integer,
            validation_steps = as_integer,
            sample_weight = as_array,
            class_weight = as_class_weight,
            verbose = as_model_verbose_arg
        ),
        ignore = c("object", "class_names", "view_metrics"),
        force = "verbose"
    )

    if (identical(view_metrics, "auto"))
      view_metrics <- resolve_view_metrics(
        args$verbose %||% as_model_verbose_arg(verbose),
        args$epochs %||% epochs,
        object$metrics)

    args$callbacks <- normalize_callbacks_with_metrics(
      view_metrics,
      (args$initial_epoch %||% initial_epoch),
      args$callbacks
    )

    # nameOfClass(tensorflow::tf$data$Dataset)
    if(inherits(args$x, "tensorflow.python.data.ops.dataset_ops.DatasetV2") &&
       !is.null(args$batch_size))
      stop("batch_size can not be specified with a TF Dataset")

    history <- do.call(object$fit, args)

    # convert to a keras_training history object
    history <- to_keras_training_history(history)

    # write metadata contained in history
    write_history_metadata(history)

    # return the history invisibly
    invisible(history)
}


input_data_normalizer <- function(model) {
  force(model)
  delayedAssign("dtype",
                as_r_value(py_get_attr(model, "input_dtype", silent = TRUE)) %||%
                  keras$config$floatx()
  )
  .normalize <- function(x) {
    if (is.null(x) || is_py_object(x))
      return(x)
    if (is.list(x))
      return(lapply(x, .normalize))
    if (is.function(x))
      return(as_data_generator(x, dtype))

    if (inherits(x, "factor"))
      x <- array(as.integer(x) - 1L,
                 dim = dim(x) %||% length(x))

    # only autocast to different sizes of the same type,
    # don't auto convert floats to ints, or ints to floats
    if (!(
      ( is.double(x) && grepl("float", dtype) ) ||
      ( is.integer(x) && grepl("int", dtype) )
    ))
      dtype <- NULL

    np_array(x, dtype)
  }
}



# ---- predict ----
#' Generates output predictions for the input samples.
#'
#' @details
#' Computation is done in batches. This method is designed for batch
#' processing of large numbers of inputs. It is not intended for use inside
#' of loops that iterate over your data and process small numbers of inputs
#' at a time.
#'
#' For small numbers of inputs that fit in one batch,
#' directly call the model `model$call` for faster execution, e.g.,
#' `model(x)`, or `model(x, training = FALSE)` if you have layers such as
#' `BatchNormalization` that behave differently during
#' inference.
#'
#' # Note
#' See [this FAQ entry](
#' https://keras.io/getting_started/faq/#whats-the-difference-between-model-methods-predict-and-call)
#' for more details about the difference between `Model` methods
#' `predict()` and `call()`.
#'
#' @returns
#' R array(s) of predictions.
#'
#' @param x
#' Input samples. It could be:
#' - A array (or array-like), or a list of arrays
#'     (in case the model has multiple inputs).
#' - A tensor, or a list of tensors
#'     (in case the model has multiple inputs).
#' - A TF Dataset.
#'
#' @param batch_size
#' Integer or `NULL`.
#' Number of samples per batch.
#' If unspecified, `batch_size` will default to `32`.
#' Do not specify the `batch_size` if your data is in the
#' form of a TF Dataset or a generator
#' (since they generate batches).
#'
#' @param verbose
#' `"auto"`, `0`, `1`, or `2`. Verbosity mode.
#' `0` = silent, `1` = progress bar, `2` = one line per epoch.
#' `"auto"` becomes 1 for most cases,
#' `2` if in a knitr render or running on a distributed training server.
#' Note that the progress bar is not
#' particularly useful when logged to a file,
#' so `verbose=2` is recommended when not running interactively
#' (e.g., in a production environment). Defaults to `"auto"`.
#'
#' @param steps
#' Total number of steps (batches of samples)
#' before declaring the prediction round finished.
#' Ignored with the default value of `NULL`.
#' If `x` is a TF Dataset and `steps` is `NULL`,
#' `predict()` will run until the input dataset is exhausted.
#'
#' @param callbacks
#' List of `Callback` instances.
#' List of callbacks to apply during prediction.
#'
#' @param object Keras model object
#' @param ... For forward/backward compatability.
#'
#' @export
#' @importFrom stats predict
#' @tether keras.Model.predict
#' @family model training
#' @seealso
#' + <https://keras.io/api/models/model_training_apis#predict-method>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/Model/predict>
predict.keras.src.models.model.Model <-
function (object, x, ..., batch_size = NULL,
          verbose = getOption("keras.verbose", default = "auto"), steps = NULL,
          callbacks = NULL)
{
    normalize_input_data <- input_data_normalizer(object)
    args <- capture_args(list(x = normalize_input_data,
                               batch_size = as_integer, steps = as_integer,
                               verbose = as_model_verbose_arg),
                          ignore = "object",
                          force = "verbose")

    if(inherits(args$x, "tensorflow.python.data.ops.dataset_ops.DatasetV2") &&
       !is.null(args$batch_size))
      stop("batch_size can not be specified with a TF Dataset")

    do.call(object$predict, args)
}

# ---- predict_on_batch ----
#' Returns predictions for a single batch of samples.
#'
#' @returns
#' Array(s) of predictions.
#'
#' @param object Keras model object
#'
#' @param x
#' Input data. It must be array-like.
#'
#' @export
#' @tether keras.Model.predict_on_batch
#' @family model training
#' @seealso
#' + <https://keras.io/api/models/model_training_apis#predictonbatch-method>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/Model/predict_on_batch>
predict_on_batch <-
function(object, x)
{
    object$predict_on_batch(as_array(x))
}


# ---- test_on_batch ----
#' Test the model on a single batch of samples.
#'
#' @returns
#' A scalar loss value (when no metrics),
#' or a named list of loss and metric values
#' (if there are metrics).
#'
#' @param x
#' Input data. Must be array-like.
#'
#' @param y
#' Target data. Must be array-like.
#'
#' @param sample_weight
#' Optional array of the same length as x, containing
#' weights to apply to the model's loss for each sample.
#' In the case of temporal data, you can pass a 2D array
#' with shape `(samples, sequence_length)`, to apply a different
#' weight to every timestep of every sample.
#'
# @param return_dict
# If `TRUE`, loss and metric results are returned as a
# dict, with each key being the name of the metric. If `FALSE`,
# they are returned as a list.
#'
#' @param object Keras model object
#' @param ... for forward/backward compatability
#'
#' @export
#' @tether keras.Model.test_on_batch
#' @family model training
#' @seealso
#' + <https://keras.io/api/models/model_training_apis#testonbatch-method>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/Model/test_on_batch>
test_on_batch <-
function (object, x, y = NULL, sample_weight = NULL, ...)
{
    result <- object$test_on_batch(as_array(x),
                                   as_array(y),
                                   as_array(sample_weight), ...,
                                   return_dict = TRUE)
    # if (length(result) > 1L) {
    #   result <- as.list(result)
    #   names(result) <- object$metrics_names
    # } else
    if (is_scalar(result)) {
      result <- result[[1L]]
    }
    result
}

# ---- test_on_batch ----
#' Runs a single gradient update on a single batch of data.
#'
#' @returns
#' A scalar loss value (when no metrics),
#' or a named list of loss and metric values
#' (if there are metrics).
#' The property `model$metrics_names`
#' will give you the display labels for the scalar outputs.
#'
#' @param x
#' Input data. Must be array-like.
#'
#' @param y
#' Target data. Must be array-like.
#'
#' @param sample_weight
#' Optional array of the same length as x, containing
#' weights to apply to the model's loss for each sample.
#' In the case of temporal data, you can pass a 2D array
#' with shape `(samples, sequence_length)`, to apply a different
#' weight to every timestep of every sample.
#'
#' @param class_weight
#' Optional named list mapping class indices (integers, 0-based)
#' to a weight (float) to apply to the model's loss for the samples
#' from this class during training. This can be useful to tell the
#' model to "pay more attention" to samples from an
#' under-represented class. When `class_weight` is specified
#' and targets have a rank of 2 or greater, either `y` must
#' be one-hot encoded, or an explicit final dimension of 1
#' must be included for sparse class labels.
#'
# @param return_dict
# If `True`, loss and metric results are returned as a
# dict, with each key being the name of the metric. If `False`,
# they are returned as a list.
#'
#' @param object Keras model object
#'
#' @export
#' @tether keras.Model.train_on_batch
#' @family model training
#' @seealso
#' + <https://keras.io/api/models/model_training_apis#trainonbatch-method>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/Model/train_on_batch>
train_on_batch <-
function (object, x, y = NULL, sample_weight = NULL, class_weight = NULL)
{
    result <- object$train_on_batch(as_array(x),
                                    as_array(y),
                                    as_array(sample_weight),
                                    class_weight = as_class_weight(class_weight),
                                    return_dict = TRUE)
    # if (length(result) > 1L) {
    #   result <- as.list(result)
    #   names(result) <- object$metrics_names
    # } else
    if (is_scalar(result)) {
      result <- result[[1L]]
    }

    result
}




# ---- summary ----
#' Print a summary of a Keras Model
#'
#' @param line_length
#' Total length of printed lines
#' (e.g. set this to adapt the display to different
#' terminal window sizes).
#'
#' @param positions
#' Relative or absolute positions of log elements
#' in each line. If not provided, becomes
#' `c(0.3, 0.6, 0.7, 1)`. Defaults to `NULL`.
#'
# ' @param print_fn
# ' Print function to use. By default, prints to `stdout`.
# ' It will be called on each line of the summary.
# ' You can set it to a custom function
# ' in order to capture the string summary.
#'
#' @param expand_nested
#' Whether to expand the nested models.
#' Defaults to `FALSE`.
#'
#' @param show_trainable
#' Whether to show if a layer is trainable.
#' Defaults to `FALSE`.
#'
#' @param layer_range
#' a list, tuple, or vector of 2 strings,
#' which is the starting layer name and ending layer name
#' (both inclusive) indicating the range of layers to be printed
#' in summary. It also accepts regex patterns instead of exact
#' name. In such case, start predicate will be the first element
#' it matches to `layer_range[[1]]` and the end predicate will be
#' the last element it matches to `layer_range[[1]]`.
#' By default `NULL` which considers all layers of model.
#'
#' @param object,x Keras model instance
#' @param line_length Total length of printed lines
#' @param positions Relative or absolute positions of log elements in each line.
#'   If not provided, defaults to `c(0.33, 0.55, 0.67, 1.0)`.
#' @param expand_nested Whether to expand the nested models. If not provided,
#'   defaults to `FALSE`.
#' @param show_trainable Whether to show if a layer is trainable. If not
#'   provided, defaults to `FALSE`.
#' @param compact Whether to remove white-space only lines from the model
#'   summary. (Default `TRUE`)
#' @param ... for `summary()` and `print()`, passed on to `format()`. For
#'   `format()`, passed on to `model$summary()`.
#'
#' @family model functions
#'
#' @returns `format()` returns a length 1 character vector. `print()` returns the
#'   model object invisibly. `summary()` returns the output of `format()`
#'   invisibly after printing it.
#'
#' @section Enabling color output in Knitr (RMarkdown, Quarto):
#'
#' In order to enable color output in a quarto or rmarkdown document with
#' an html output format (include revealjs presentations), then you will need
#' to do the following in a setup chunk:
#'
#'
#' ````
#'  ```{r setup, include = FALSE}
#'  options(cli.num_colors = 256)
#'  fansi::set_knit_hooks(knitr::knit_hooks)
#'  options(width = 75) # adjust as needed for format
#'  ```
#' ````
#'
#'
#' @export
summary.keras.src.models.model.Model <- function(object, ...) {
    writeLines(f <- format.keras.src.models.model.Model(object, ...))
    invisible(f)
}


#' @rdname summary.keras.src.models.model.Model
#' @export
format.keras.src.models.model.Model <-
function(x,
         line_length = getOption("width"), # width - (12L * show_trainable),
         positions = NULL,
         expand_nested = FALSE,
         show_trainable = NA,
         ...,
         # TODO: add force_ascii arg
         # force_ascii ... (impl in man/roxygen/meta.R)
         # width = getOption("width"),
         # rich = TRUE, ??
         # print_fn = NULL,
         layer_range = NULL,
         compact = TRUE) {

    if (py_is_null_xptr(x))
        return("<pointer: 0x0>")

    args <- capture_args(ignore = c("x", "compact", "width"),
                          force = c("show_trainable", "line_length"))

    if(is.na(args$show_trainable)) {
      built <- as_r_value(py_get_attr(x, "built", silent = TRUE)) %||% FALSE
      args$show_trainable <- built && as.logical(length(x$non_trainable_weights))
    }

    with_rich_config(
      out <- trimws(py_capture_output(do.call(x$summary, args)))
    )

    if(compact) {
        # strip empty lines
        out <- gsub("(\\n\\s*\\n)", "\n", out, perl = TRUE)
        if(expand_nested)
            out <- gsub("\\n\\|\\s+\\|\\n", "\n", out)
    }

    out
}

#
#' @rdname summary.keras.src.models.model.Model
#' @export
print.keras.src.models.model.Model <- function(x, ...) {
    writeLines(format.keras.src.models.model.Model(x, ...))
    invisible(x)
}

#' @importFrom reticulate py_str
#' @export
py_str.keras.src.models.model.Model <- function(object, ...) {
    format.keras.src.models.model.Model(object, ...)
}


with_rich_config <- function(expr) {

  vars <- list(
    COLUMNS = as.character(getOption("width"))
  )

  if (Sys.getenv("COLORTERM", "truecolor") == "truecolor" &&
      cli::num_ansi_colors() >= 256L) {
    vars$COLORTERM <- "truecolor"
    vars$FORCE_COLOR <- "yes"
  }

  with_envvar2(vars, expr)
}


with_envvar2 <- function(vars, expr) {
  py_environ <- import("os", convert = FALSE)$environ

  og_r_vars <- Sys.getenv(names(vars), unset = NA_character_, names = TRUE)
  og_py_vars <- lapply(names(vars), function(key)
    py_get_item(py_environ, key, silent = TRUE))
  names(og_py_vars) <- names(vars)

  names_unset_vars <-
    names(vars[map_lgl(vars, function(v) is.null(v) || is.na(v))])
  vars <- vars[setdiff(names(vars), names_unset_vars)]
  if (length(vars)) {
    do.call(Sys.setenv, as.list(vars))
    imap(vars, function(val, key) {
      py_set_item(py_environ, key, val)
    })
  }
  for (name in names_unset_vars) {
    Sys.unsetenv(name)
    py_del_item(py_environ, name)
  }

  on.exit({
    og_r_var_was_unset <- is.na(og_r_vars)
    set_r_vars <- og_r_vars[!og_r_var_was_unset]
    if (length(set_r_vars))
      do.call(Sys.setenv, as.list(set_r_vars))
    for (name in names(og_r_vars)[og_r_var_was_unset])
      Sys.unsetenv(name)

    imap(og_py_vars, function(val, key) {
      if (is.null(val))
        py_del_item(py_environ, key)
      else
        py_set_item(py_environ, key, val)
      NULL
    })

    NULL
  }, add = TRUE)
  force(expr)
}



# ---- internal utils ----



as_model_verbose_arg <- function(x) {
  if(!identical(x, "auto"))
    return(as.integer(x))
  # x == auto
  if(isTRUE(getOption('knitr.in.progress')))
    return(2L)
  x # "auto"
}


as_class_weight <- function(class_weight, class_names = NULL) {
  if (is.null(class_weight))
    return(NULL)
  if (is.numeric(class_weight))
    class_weight <- as.list(class_weight)

  # convert class weights to python dict
  if (is.list(class_weight))
    # dict() converts numeric (chr) names to numeric (dbl) keys
    return(dict(class_weight))

  stop("class_weight must be a named list of weights")
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



py_generator <- function(fn, completed = NULL, prefetch = 0L, convert = FALSE) {
  iterator2generator <- py_eval("lambda iterator: (yield from iterator)",
                                convert = convert)
  py_call(iterator2generator, py_iterator(fn, completed, prefetch))
}


as_data_generator <- function(fn, dtype = NULL) {
  force(fn); force(dtype)
  python_path <- system.file("python", package = "keras3")
  tools <- reticulate::import_from_path("kerastools", path = python_path)

  py_generator(function() {
    x <- keras_array(fn(), dtype = dtype)
    if (is.null(x))
      NULL
    else
      tuple(x)
  }, completed = NULL, prefetch = 1L)

}






# ' @exportS3Method knitr::knit_print
knit_print__keras.src.models.model.Model <- function(x, ...) {
  #from keras.src.utils.summary_utils
  # record_console <- py_run_string(local = TRUE, glue::trim("
  # class record_console:
  #   def __init__(self):
  #     self.last_console = None
  #
  #   def __enter__(self, *args):
  #     import rich
  #     self.rich = rich
  #     from functools import wraps
  #     og_Console =
  #     self.og_Console = rich.console.Console
  #     @wraps(og_Console)
  #     def Console(*args, record = True, **kwargs):
  #         kwargs['record'] = record
  #         global last_console
  #         self.last_console = self.og_Console(*args, **kwargs)
  #         return self.last_console
  #     rich.console.Console = Console
  #
  #   def __exit__(self, *args):
  #       self.rich.console.Console = self.og_Console
  #   "))$record_console

  knitrtools <- import_kerastools("knitr")
  recorder <- knitrtools$RichConsoleRecorder()
  # restore <- py_local$restore
  with(recorder, {
    format.keras.src.models.model.Model(x)
  })

  if(knitr::is_html_output()) {
    html <- recorder$console$export_html(
      inline_styles = TRUE,
      clear = TRUE
    )
    knitr::raw_html(html)
  } else {
    text <- recorder$console$export_text(
      styles = FALSE, # plain text
      clear = TRUE
    )

    text
  }

}
