

# ---- core ----


#' Create a Keras Layer
#'
#' @param layer_class A Python Layer class
#' @param object Object to compose layer with. This is either a
#' [keras_model_sequential()] to add the layer to, or another Layer which
#' this layer will call.
#' @param args List of arguments to the layer initialize function.
#'
#' @returns A Keras layer
#'
#' @note The `object` parameter can be missing, in which case the
#' layer is created without a connection to an existing graph.
#'
#' @keywords internal
#' @noRd
create_layer <- function(LayerClass, object, args = list()) {

  # force `object` before instantiating the layer, so pipe chains create layers
  # in the the intutitively expected order.
  # https://github.com/rstudio/keras/issues/1440
  object <- if (missing(object)) NULL else object

  # Starting in Keras 3.1, constraints can't be simple callable functions, they
  # *must* inherit from keras.constraints.Constraint()
  args <- imap(args, function(arg, name) {
    if (endsWith(name, "_constraint") && is_bare_r_function(arg))
      arg <- as_constraint(arg)
    arg
  })

  args <- lapply(args, resolve_py_obj)

  if (!is_py_object(LayerClass)) # e.g., R6ClassGenerator
    LayerClass <- r_to_py(LayerClass)

  # create layer instance by calling the LayerClass object
  layer <- do.call(LayerClass, args)

  # compose if we have an `object`
  if (is.null(object))
    layer
  else
    compose_layer(object, layer)
}


# Helper function to enable composing a layer instance with a Sequential model
# via a simple call like layer(<sequential_model>).
compose_layer <- function(object, layer, ...) {
  if(missing(object) || is.null(object))
    return(layer(...))

  # if the first arg is a Sequential model, call `model$add()`
  if (inherits(object, "keras.src.models.sequential.Sequential")) {
    if(length(list(...)) > 0) warning("arguments passed via ellipsis will be ignored")

    object$add(layer)
    return(invisible(object))
  }

  # otherwise, invoke `layer$__call__()`
  layer(object, ...)
}


# TODO: use formals(x) in py_to_r_wrapper.Layer() to construct a better wrapper fn
# (( though, all layer.__call__ signatures are generally (...), unless user
#     implemented __call__() directly instead of call() ))

# This is used for:
# - ALL layer instances (custom and builtin) and
# - ALL model instances (Sequential, Functional, and custom)
#' @export
py_to_r_wrapper.keras.src.layers.layer.Layer <- function(x) {
  force(x)
  function(object, ...) compose_layer(object = object, layer = x, ...)
}


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
