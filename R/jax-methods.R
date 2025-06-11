
#' @exportS3Method as.array jax.Array
as.array.jax.Array <- function(x, ...) {
  import("numpy")$asarray(x)
}

#' @exportS3Method as.array jaxlib._jax.ArrayImpl
as.array.jaxlib._jax.ArrayImpl <- as.array.jax.Array

#' @exportS3Method as.array jaxlib.xla_extension.ArrayImpl
as.array.jaxlib.xla_extension.ArrayImpl <- as.array.jax.Array


#' @exportS3Method as.double jax.Array
as.double.jax.Array <- function(x, ...) {
  as.double(import("numpy")$asarray(x))
}

#' @exportS3Method as.double jaxlib._jax.ArrayImpl
as.double.jaxlib._jax.ArrayImpl <- as.double.jax.Array

#' @exportS3Method as.double jaxlib.xla_extension.ArrayImpl
as.double.jaxlib.xla_extension.ArrayImpl <- as.double.jax.Array


#' @exportS3Method as.integer jax.Array
as.integer.jax.Array <- function(x, ...) {
  as.integer(import("numpy")$asarray(x))
}

#' @exportS3Method as.integer jaxlib._jax.ArrayImpl
as.integer.jaxlib._jax.ArrayImpl <- as.integer.jax.Array

#' @exportS3Method as.integer jaxlib.xla_extension.ArrayImpl
as.integer.jaxlib.xla_extension.ArrayImpl <- as.integer.jax.Array


#' @exportS3Method as.numeric jax.Array
as.numeric.jax.Array <- function(x, ...) {
  as.numeric(import("numpy")$asarray(x))
}

#' @exportS3Method as.numeric jaxlib._jax.ArrayImpl
as.numeric.jaxlib._jax.ArrayImpl <- as.numeric.jax.Array

#' @exportS3Method as.numeric jaxlib.xla_extension.ArrayImpl
as.numeric.jaxlib.xla_extension.ArrayImpl <- as.numeric.jax.Array


#' @exportS3Method str jax.Array
str.jax.Array <- function(x, ...) {
  shape <- py_to_r(x$shape)
  shape <- unlist(lapply(shape, function(axis) {
    if (is.null(axis)) NA_integer_ else as.integer(axis)
  }))
  shape <- paste0(as.integer(shape), collapse = ", ")
  dtype <- as.character(py_to_r(x$dtype$name))
  cat(sep = "",
      if (nest.lev > 0) " ",
      sprintf("<jax.Array shape(%s), dtype=%s>\n", shape, dtype))
}

#' @exportS3Method str jaxlib._jax.ArrayImpl
str.jaxlib._jax.ArrayImpl <- str.jax.Array

#' @exportS3Method str jaxlib.xla_extension.ArrayImpl
str.jaxlib.xla_extension.ArrayImpl <- str.jax.Array

#' @exportS3Method str keras.src.backend.jax.core.Variable
str.keras.src.backend.jax.core.Variable <- function(object, ...) {
  writeLines(type_sum.keras.src.backend.jax.core.Variable(object))
}

#' @exportS3Method pillar::type_sum keras.src.backend.jax.core.Variable
type_sum.keras.src.backend.jax.core.Variable <- function(x) {
  if (reticulate::py_is_null_xptr(x))
    return("<pointer: 0x0>")
  x <- reticulate::py_repr(x)
  x <- strsplit(x, "\n", fixed = TRUE)[[1L]]
  if (length(x) > 1L || nchar(x) > getOption("width")) {
    x <- sub("(value=.+)", "value=[…]>", x[1L])
  }
  # compact shapes like (None, 10) to (None) for readability
  x <- sub("shape=\\((None|[[:digit:]]+),\\)", "shape=(\\1)", x)
  x
}
