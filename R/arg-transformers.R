
as_axis <- function(axis) {
  if (is.null(axis))
    return(NULL)

  if (length(axis) > 1)
    return(lapply(axis, as_axis))

  axis <- as.integer(axis)

  if (axis == 0L)
    stop("`axis` argument is 1 based, received 0")

  if (axis > 0L) axis - 1L
  else axis
}

# Helper function to coerce shape arguments to tuple
# tf$reshape()/k_reshape() doesn't accept a tf.TensorShape object
normalize_shape <- function(shape) {

  # reflect NULL back
  if (is.null(shape))
    return(shape)

  # already fixed up
  if(inherits(shape, "keras_shape"))
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

as_integer <- function(x) {
  if (is.numeric(x))
    as.integer(x)
  else
    x
}

as_integer_array <- function(x) {
  if(is.atomic(x))
    x <- as.array(x)
  if(is.array(x) && storage.mode(x) != "integer")
    storage.mode(x) <- "integer"
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



# unused
as_index <- function(x) {
  if(storage.mode(x) == "double")
    storage.mode(x) <- "integer"
  # k_array() pass through here...
  # TODO: implement an efficient way to check for negative slices
  x - 1L
}
