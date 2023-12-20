

# ---- core ----


#' Create a Keras Layer
#'
#' @param layer_class Python layer class or R6 class of type KerasLayer
#' @param object Object to compose layer with. This is either a
#' [keras_model_sequential()] to add the layer to, or another Layer which
#' this layer will call.
#' @param args List of arguments to layer constructor function
#'
#' @return A Keras layer
#'
#' @note The `object` parameter can be missing, in which case the
#' layer is created without a connection to an existing graph.
#'
#' @export
create_layer <- function(layer_class, object, args = list()) {

  safe_to_drop_nulls <- c(
    "input_shape",
    "batch_input_shape",
    "batch_size",
    "dtype",
    "name",
    "trainable",
    "weights"
  )
  for (nm in safe_to_drop_nulls)
    args[[nm]] <- args[[nm]]

  # convert custom constraints
  constraint_args <- grepl("^.*_constraint$", names(args))
  constraint_args <- names(args)[constraint_args]
  for (arg in constraint_args)
    args[[arg]] <- as_constraint(args[[arg]])

  if (inherits(layer_class, "R6ClassGenerator")) {

    if (identical(layer_class$get_inherit(), KerasLayer)) {
      # old-style custom class, inherits KerasLayer
      c(layer, args) %<-% compat_custom_KerasLayer_handler(layer_class, args)
      layer_class <- function(...) layer
    } else {
      # new-style custom class, inherits anything else, typically keras$layers$Layer
      layer_class <- r_to_py(layer_class, convert = TRUE)
    }
  }

  # create layer from class
  layer <- do.call(layer_class, args)

  # compose if we have an x
  if (missing(object) || is.null(object))
    layer
  else
    invisible(compose_layer(object, layer))
}


# Helper function to compose a layer with an object of type Model or Layer

compose_layer <- function(object, layer, ...) {
  UseMethod("compose_layer")
}

#' @export
compose_layer.default <- function(object, layer, ...) {
  layer(object, ...)
}

#' @export
compose_layer.keras.models.Sequential <- function(object, layer, ...) {
  if(length(list(...)) > 0) warning("arguments passed via ellipsis will be ignored")

  object$add(layer)
  object
}

compose_layer.keras.engine.sequential.Sequential <- compose_layer.keras.models.Sequential
compose_layer.keras.models.sequential.Sequential <- compose_layer.keras.models.Sequential

# compose_layer.keras.src.engine.sequential.Sequential <- compose_layer.keras.models.Sequential



# ---- convolutional ----
normalize_padding <- function(padding, dims) {
  normalize_scale("padding", padding, dims)
}

normalize_cropping <- function(cropping, dims) {
  normalize_scale("cropping", cropping, dims)
}

normalize_scale <- function(name, scale, dims) {

  # validate and marshall scale argument
  throw_invalid_scale <- function() {
    stop(name, " must be a list of ", dims, " integers or list of ", dims,  " lists of 2 integers",
         call. = FALSE)
  }

  # if all of the individual items are numeric then cast to integer vector
  if (all(sapply(scale, function(x) length(x) == 1 && is.numeric(x)))) {
    as.integer(scale)
  } else if (is.list(scale)) {
    lapply(scale, function(x) {
      if (length(x) != 2)
        throw_invalid_scale()
      as.integer(x)
    })
  } else {
    throw_invalid_scale()
  }
}



# ---- preprocessing ----




# ---- text preprocessing ----



#' @param include_special_tokens If TRUE, the returned vocabulary will include
#'   the padding and OOV tokens, and a term's index in the vocabulary will equal
#'   the term's index when calling the layer. If FALSE, the returned vocabulary
#'   will not include any padding or OOV tokens.
#' @rdname layer_text_vectorization
#' @export
get_vocabulary <- function(object, include_special_tokens=TRUE) {
  if (tensorflow::tf_version() < "2.3") {
    python_path <- system.file("python", package = "keras3")
    tools <- import_from_path("kerastools", path = python_path)
    tools$get_vocabulary$get_vocabulary(object)
  } else {
    args <- capture_args(match.call(), ignore = "object")
    do.call(object$get_vocabulary, args)
  }
}

#' @rdname layer_text_vectorization
#' @param idf_weights An R vector, 1D numpy array, or 1D tensor of inverse
#'   document frequency weights with equal length to vocabulary. Must be set if
#'   output_mode is "tf_idf". Should not be set otherwise.
#' @export
set_vocabulary <- function(object, vocabulary, idf_weights=NULL, ...) {
  args <- capture_args(match.call(), ignore = "object")

  if (tf_version() < "2.6") {
    # required arg renamed when promoted out of experimental in 2.6:
    #   vocab -> vocabulary. Position as first unchanged.
    vocab <- args[["vocabulary"]] %||% args[["vocab"]]
    args[c("vocab", "vocabulary")] <- NULL
    args <- c(list(vocab), args)
  }

  do.call(object$set_vocabulary, args)
  invisible(object)
}



#' Fits the state of the preprocessing layer to the data being passed
#'
#' @details
#' After calling `adapt` on a layer, a preprocessing layer's state will not
#' update during training. In order to make preprocessing layers efficient in
#' any distribution context, they are kept constant with respect to any
#' compiled `tf.Graph`s that call the layer. This does not affect the layer use
#' when adapting each layer only once, but if you adapt a layer multiple times
#' you will need to take care to re-compile any compiled functions as follows:
#'
#'  * If you are adding a preprocessing layer to a `keras.Model`, you need to
#'    call `compile(model)` after each subsequent call to `adapt()`.
#'  * If you are calling a preprocessing layer inside `tfdatasets::dataset_map()`,
#'    you should call `dataset_map()` again on the input `tf.data.Dataset` after each
#'    `adapt()`.
#'  * If you are using a `tensorflow::tf_function()` directly which calls a preprocessing
#'    layer, you need to call `tf_function` again on your callable after
#'    each subsequent call to `adapt()`.
#'
#' `keras_model` example with multiple adapts:
#' ````r
#' layer <- layer_normalization(axis=NULL)
#' adapt(layer, c(0, 2))
#' model <- keras_model_sequential(layer)
#' predict(model, c(0, 1, 2)) # [1] -1  0  1
#'
#' adapt(layer, c(-1, 1))
#' compile(model)  # This is needed to re-compile model.predict!
#' predict(model, c(0, 1, 2)) # [1] 0 1 2
#' ````
#'
#' `tf.data.Dataset` example with multiple adapts:
#' ````r
#' layer <- layer_normalization(axis=NULL)
#' adapt(layer, c(0, 2))
#' input_ds <- tfdatasets::range_dataset(0, 3)
#' normalized_ds <- input_ds %>%
#'   tfdatasets::dataset_map(layer)
#' str(reticulate::iterate(normalized_ds))
#' # List of 3
#' #  $ :tf.Tensor([-1.], shape=(1,), dtype=float32)
#' #  $ :tf.Tensor([0.], shape=(1,), dtype=float32)
#' #  $ :tf.Tensor([1.], shape=(1,), dtype=float32)
#' adapt(layer, c(-1, 1))
#' normalized_ds <- input_ds %>%
#'   tfdatasets::dataset_map(layer) # Re-map over the input dataset.
#' str(reticulate::iterate(normalized_ds$as_numpy_iterator()))
#' # List of 3
#' #  $ : num [1(1d)] -1
#' #  $ : num [1(1d)] 0
#' #  $ : num [1(1d)] 1
#' ````
#'
#' @param object Preprocessing layer object
#'
#' @param data The data to train on. It can be passed either as a
#'   `tf.data.Dataset` or as an R array.
#'
#' @param batch_size Integer or `NULL`. Number of asamples per state update. If
#'   unspecified, `batch_size` will default to `32`. Do not specify the
#'   batch_size if your data is in the form of datasets, generators, or
#'   `keras.utils.Sequence` instances (since they generate batches).
#'
#' @param steps Integer or `NULL`. Total number of steps (batches of samples)
#'   When training with input tensors such as TensorFlow data tensors, the
#'   default `NULL` is equal to the number of samples in your dataset divided by
#'   the batch size, or `1` if that cannot be determined. If x is a
#'   `tf.data.Dataset`, and `steps` is `NULL`, the epoch will run until the
#'   input dataset is exhausted. When passing an infinitely repeating dataset,
#'   you must specify the steps argument. This argument is not supported with
#'   array inputs.
#'
#' @param ... Used for forwards and backwards compatibility. Passed on to the underlying method.
#'
#' @family preprocessing layer methods
#'
#' @seealso
#'
#'  + <https://www.tensorflow.org/guide/keras/preprocessing_layers#the_adapt_method>
#'  + <https://keras.io/guides/preprocessing_layers/#the-adapt-method>
#'
#' @export
adapt <- function(object, data, ..., batch_size=NULL, steps=NULL) {
  if (!inherits(data, "python.builtin.object"))
    data <- keras_array(data)
  # TODO: use as_tensor() here

  args <- capture_args(match.call(),
    list(batch_size = as_nullable_integer,
         step = as_nullable_integer),
    ignore = c("object", "data"))
  # `data` named to `dataset` in keras3 keras.utils.FeatureSpace
  # pass it as a positional arg
  args <- c(list(data), args)
  do.call(object$adapt, args)
  invisible(object)
}





## TODO: TextVectorization has a compile() method. investigate if this is
## actually useful to export
#compile.keras.engine.base_preprocessing_layer.PreprocessingLayer <-
function(object, run_eagerly = NULL, steps_per_execution = NULL, ...) {
  args <- capture_args(match.call(), ignore="object")
  do.call(object$compile, args)
}



# TODO: add an 'experimental' tag in the R docs where appropriate

require_tf_version <- function(ver, msg = "this function.") {
  if (tf_version() < ver)
    stop("Tensorflow version >=", ver, " required to use ", msg)
}


# in python keras, sometimes strings are compared with `is`, which is too strict
# https://github.com/keras-team/keras/blob/db3fa5d40ed19cdf89fc295e8d0e317fb64480d4/keras/layers/preprocessing/text_vectorization.py#L524
# # there was already 1 PR submitted to fix this:
# https://github.com/tensorflow/tensorflow/pull/34420
# This is a hack around that: deparse and reparse the string in python. This
# gives the python interpreter a chance to recycle the address for the identical
# string that's already in it's string pool, which then passes the `is`
# comparison.
#TODO: delete this
fix_string <- local({
  py_reparse <- NULL # R CMD check: no visible binding
  delayedAssign("py_reparse",
    py_eval("lambda x: eval(repr(x), {'__builtins__':{}}, {})",
            convert = FALSE))
  # Note, the globals dict {'__builtins__':{}} is a guardrail, not a security
  # door. A well crafted string can still break out to execute arbitrary code in
  # the python session, but it can do no more than can be done from the R
  # process already.
  # Can't use ast.literal_eval() because it fails the 'is' test. E.g.:
  # bool('foo' is ast.literal_eval("'foo'")) == FALSE
  function(x) {
    if (is.character(x))
      py_call(py_reparse, as.character(x))
    else
      x
  }
})


# ---- wrappers ----

#' This layer wrapper allows to apply a layer to every temporal slice of an input
#'
#' @details
#' Every input should be at least 3D, and the dimension of index one of the
#' first input will be considered to be the temporal dimension.
#'
#' Consider a batch of 32 video samples, where each sample is a 128x128 RGB image
#' with `channels_last` data format, across 10 timesteps.
#' The batch input shape is `(32, 10, 128, 128, 3)`.
#'
#' You can then use `TimeDistributed` to apply the same `Conv2D` layer to each
#' of the 10 timesteps, independently:
#'
#' ```R
#' input <- layer_input(c(10, 128, 128, 3))
#' conv_layer <- layer_conv_2d(filters = 64, kernel_size = c(3, 3))
#' output <- input %>% time_distributed(conv_layer)
#' shape(output) # shape(NA, 10, 126, 126, 64)
#' ```
#'
#' Because `TimeDistributed` applies the same instance of `Conv2D` to each of the
#' timestamps, the same set of weights are used at each timestamp.
#'
#' @inheritParams layer_dense
#' @param layer a `tf.keras.layers.Layer` instance.
#' @param ... standard layer arguments.
#'
#' @seealso
#'   +  <https://www.tensorflow.org/api_docs/python/tf/keras/layers/TimeDistributed>
#'
#' @family layer wrappers
#' @export
time_distributed <-
function(object, layer, ...)
{
  args <-
    capture_args(
      match.call(),
      list(
        input_shape = normalize_shape,
        batch_input_shape = normalize_shape,
        batch_size = as_nullable_integer
      ),
      ignore = "object"
    )
    create_layer(keras$layers$TimeDistributed, object, args)
}


#  Bidirectional wrapper for RNNs
#
#  @inheritParams layer_dense
#
#  @param layer A `RNN` layer instance, such as `layer_lstm()` or
#    `layer_gru()`. It could also be a `keras$layers$Layer` instance that
#    meets the following criteria:
#
#    1. Be a sequence-processing layer (accepts 3D+ inputs).
#
#    2. Have a `go_backwards`, `return_sequences` and `return_state` attribute
#    (with the same semantics as for the `RNN` class).
#
#    3. Have an `input_spec` attribute.
#
#    4. Implement serialization via `get_config()` and `from_config()`. Note
#    that the recommended way to create new RNN layers is to write a custom RNN
#    cell and use it with `layer_rnn()`, instead of subclassing
#    `keras$layers$Layer` directly.
#
#    5. When `returns_sequences = TRUE`, the output of the masked timestep will
#    be zero regardless of the layer's original `zero_output_for_mask` value.
#
#  @param merge_mode Mode by which outputs of the forward and backward RNNs will
#    be combined. One of `'sum'`, `'mul'`, `'concat'`, `'ave'`, `NULL`. If
#    `NULL`, the outputs will not be combined, they will be returned as a list.
#    Default value is `'concat'`.
#
#  @param weights Split and propagated to the `initial_weights` attribute on the
#    forward and backward layer.
#
#  @param backward_layer Optional `keras.layers.RNN`, or `keras.layers.Layer`
#    instance to be used to handle backwards input processing. If
#    `backward_layer` is not provided, the layer instance passed as the `layer`
#    argument will be used to generate the backward layer automatically. Note
#    that the provided `backward_layer` layer should have properties matching
#    those of the `layer` argument, in particular it should have the same values
#    for `stateful`, `return_states`, `return_sequences`, etc. In addition,
#    `backward_layer` and `layer` should have different `go_backwards` argument
#    values. A `ValueError` will be raised if these requirements are not met.
#
#  @param ... standard layer arguments.
#
#  @family layer wrappers
#  @seealso
#
#  - <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Bidirectional>
#  - <https://keras.io/api/layers/recurrent_layers/bidirectional/>
#
#  @export
# bidirectional <-
function(object, layer, merge_mode = "concat",
         weights = NULL, backward_layer = NULL, ...)
{
  args <- capture_args(
    match.call(),
    modifiers = list(
      input_shape = normalize_shape,
      batch_input_shape = normalize_shape,
      batch_size = as_nullable_integer
    ),
    ignore = "object"
  )
  create_layer(keras$layers$Bidirectional, object, args)
}
