


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

as_index <- function(x) {
  if(storage.mode(x) == "double")
    storage.mode(x) <- "integer"
  # k_array() pass through here...
  # TODO: implement an efficient way to check for negative slices
  x - 1L
}


backend_normalize_shape <- function(shape) {

  # if it's a Python object or a list with python objects then leave it alone
  if (inherits(shape, "python.builtin.object"))
    return(shape)

  normalize_shape(shape)
}
