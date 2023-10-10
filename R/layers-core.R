

as_integer <- function(x) {
  if (is.numeric(x))
    as.integer(x)
  else
    x
}

as_integer_tuple <- function(x, force_tuple = FALSE) {
  if (is.null(x))
    x
  else if (is.list(x) || force_tuple)
    tuple(as.list(as.integer(x)))
  else
    as.integer(x)
}

as_nullable_integer <- function(x) {
  if (is.null(x))
    x
  else
    as.integer(x)
}

as_layer_index <- function(x) {
  if (is.null(x))
    return(x)

  x <- as.integer(x)

  if (x == 0L)
    stop("`index` for get_layer() is 1-based (0 was passed as the index)")

  if (x > 0L)
    x - 1L
  else
    x
}

# Helper function to normalize paths
normalize_path <- function(path) {
  if (is.null(path))
    NULL
  else
    normalizePath(path.expand(path), mustWork = FALSE)
}



# Helper function to coerce shape arguments to tuple
# tf$reshape()/k_reshape() doesn't accept a tf.TensorShape object
normalize_shape <- function(shape) {

  # reflect NULL back
  if (is.null(shape))
    return(shape)

  # if it's a list or a numeric vector then convert to integer
  # NA's in are accepted as NULL
  # also accept c(NA), as if it was a numeric
  if (is.list(shape) || is.numeric(shape) ||
      (is.logical(shape) && all(is.na(shape)))) {

    shape <- lapply(shape, function(value) {
      # Pass through python objects unmodified, only coerce R objects
      # supplied shapes, e.g., to tf$random$normal, can be a list that's a mix
      # of scalar integer tensors and regular integers
      if (inherits(value, "python.builtin.object"))
        return(value)

      # accept NA,NA_integer_,NA_real_ as NULL
      if ((is_scalar(value) && is.na(value)))
        return(NULL)

      if (!is.null(value))
        as.integer(value)
      else
        NULL
    })
  }

  if (inherits(shape, "tensorflow.python.framework.tensor_shape.TensorShape"))
    shape <- as.list(shape$as_list()) # unpack for tuple()

  # coerce to tuple so it's iterable
  tuple(shape)
}

# @export
# format.python.builtin.object <- function(x, ...) {
#   capture.output(print(x, ...))
# }

as_shape <- function(x) {
  lapply(x, function(d) {
    if (is.null(d))
      NULL
    else
      as.integer(d)
  })
}

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

compose_layer.default <- function(object, layer, ...) {
  layer(object, ...)
}

compose_layer.keras.models.Sequential <- function(object, layer, ...) {
  if(length(list(...)) > 0) warning("arguments passed via ellipsis will be ignored")

  object$add(layer)
  object
}

compose_layer.keras.engine.sequential.Sequential <- compose_layer.keras.models.Sequential
compose_layer.keras.models.sequential.Sequential <- compose_layer.keras.models.Sequential

# compose_layer.keras.src.engine.sequential.Sequential <- compose_layer.keras.models.Sequential
