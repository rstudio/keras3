

# ---- core ----


#' Create a Keras Layer
#'
#' @param layer_class A Python Layer class
#' @param object Object to compose layer with. This is either a
#' [keras_model_sequential()] to add the layer to, or another Layer which
#' this layer will call.
#' @param args List of arguments to the layer initialize function.
#'
#' @return A Keras layer
#'
#' @note The `object` parameter can be missing, in which case the
#' layer is created without a connection to an existing graph.
#'
#' @keywords internal
#' @noRd
create_layer <- function(layer_class, object, args = list()) {

  args <- lapply(args, resolve_py_obj)

  if (!inherits(layer_class, "python.builtin.object")) # e.g., R6ClassGenerator
    layer_class <- r_to_py(layer_class)

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
  if(missing(object) || is.null(object))
    return(layer(...))

  # if the first arg is a Sequential model, call `model$add()`
  if (inherits(object, "keras.src.models.sequential.Sequential")) {
    if(length(list(...)) > 0) warning("arguments passed via ellipsis will be ignored")

    object$add(layer)
    return(object)
  }

  # otherwise, invoke `layer$__call__()`
  layer(object, ...)
}

# compose_layer2 <- function(object, layer, other_call_args = list()) {
#   if(missing(object) || is.null(object))
#     return(do.call(layer, other_call_args))
#
#   # if the first arg is a Sequential model, call `model$add()`
#   if (inherits(object, "keras.src.models.sequential.Sequential")) {
#     if(length(other_call_args) > 0) warning("arguments passed via ellipsis will be ignored")
#
#     object$add(layer)
#     return(object)
#   }
#
#   # otherwise, invoke `layer$__call__()`
#   do.call(layer, c(object, other_call_args))
# }

# TODO: use formals(x) in py_to_r_wrapper() to construct a better wrapper fn
# This is used for ALL layers (custom, and builtin)
#' @export
py_to_r_wrapper.keras.src.layers.layer.Layer <- function(x) {
  force(x)
  function(object, ...) compose_layer(object = object, layer = x, ...)
}

# py_to_r_wrapper.keras.src.layers.layer.Layer <- function(x) {
#   force(x)
#   build_layer_instance_composing_wrapper(x)
# }

# build_layer_instance_composing_wrapper <- function(x, envir = parent.frame()) {
#   arg1 <- as.symbol(names(formals(x))[[1]])
#   args_rest <- quote(list(...))
#   if(identical(arg1, quote(...))) {
#     arg1 <- quote(..1)
#     args_rest <- quote(list(...)[-1])
#   }
#   bdy <- bquote(compose_layer(object = .(arg1), layer = x, ...))
# }


# models are layers, the layer wrapper should suffice...
# TODO: delete this after confirming not breaking
# ' @importFrom reticulate py_to_r_wrapper
# ' @export
# py_to_r_wrapper.keras.src.models.model.Model <- function(x) {
#   force(x)
#   function(object, ...) {
#     compose_layer(object, x, ...)
#   }
# }



#  py_to_r_wrapper.keras.engine.base_layer.Layer <- function(x) {
#    force(x)
#    function(...) {
#      if(!missing(..1) && inherits(..1, "keras.src..."   "keras.engine.sequential.Sequential")) {
#        if(length(list(...)) > 1)
#          warning("Other arguments to ... are ignored because layer instance already created")
#        model <- ..1
#        model$add(x)
#        model
#      } else
#        x(...)
#    }
#  }




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





# TODO: add an 'experimental' tag in the R docs where appropriate

require_tf_version <- function(ver, msg = "this function.") {
  if (tf_version() < ver)
    stop("Tensorflow version >=", ver, " required to use ", msg)
}



# ---- wrappers ----

#  This layer wrapper allows to apply a layer to every temporal slice of an input
#
#  @details
#  Every input should be at least 3D, and the dimension of index one of the
#  first input will be considered to be the temporal dimension.
#
#  Consider a batch of 32 video samples, where each sample is a 128x128 RGB image
#  with `channels_last` data format, across 10 timesteps.
#  The batch input shape is `(32, 10, 128, 128, 3)`.
#
#  You can then use `TimeDistributed` to apply the same `Conv2D` layer to each
#  of the 10 timesteps, independently:
#
#  ```R
#  input <- layer_input(c(10, 128, 128, 3))
#  conv_layer <- layer_conv_2d(filters = 64, kernel_size = c(3, 3))
#  output <- input %>% time_distributed(conv_layer)
#  shape(output) # shape(NA, 10, 126, 126, 64)
#  ```
#
#  Because `TimeDistributed` applies the same instance of `Conv2D` to each of the
#  timestamps, the same set of weights are used at each timestamp.
#
#  @inheritParams layer_dense
#  @param layer a `tf.keras.layers.Layer` instance.
#  @param ... standard layer arguments.
#
#  @seealso
#    +  <https://www.tensorflow.org/api_docs/python/tf/keras/layers/TimeDistributed>
#
#  @family layer wrappers
#  @export
# time_distributed <-
function(object, layer, ...)
{
  args <- capture_args(list(input_shape = normalize_shape,
                             batch_input_shape = normalize_shape,
                             batch_size = as_nullable_integer),
                        ignore = "object")
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
    modifiers = list(
      input_shape = normalize_shape,
      batch_input_shape = normalize_shape,
      batch_size = as_nullable_integer
    ),
    ignore = "object"
  )
  create_layer(keras$layers$Bidirectional, object, args)
}
