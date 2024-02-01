
#' @export
`==.keras.src.backend.common.keras_tensor.KerasTensor` <- function(e1, e2) {
  op_equal(e1, e2)
}

#' @export
`+.keras.src.backend.common.keras_tensor.KerasTensor` <- function(e1, e2) {
  if(missing(e2)) return(e1)
  NextMethod()
}


#' @export
as.array.keras.src.backend.common.variables.KerasVariable <- function(x, ...) {
  as_r_value(keras$ops$convert_to_numpy(x))
}

#' @export
as.numeric.keras.src.backend.common.variables.KerasVariable <- function(x, ...) {
  as.numeric(as_r_value(keras$ops$convert_to_numpy(x)))
}

#' @export
as.double.keras.src.backend.common.variables.KerasVariable <- function(x, ...) {
  as.double(as_r_value(keras$ops$convert_to_numpy(x)))
}

#' @export
as.integer.keras.src.backend.common.variables.KerasVariable <- function(x, ...) {
  as.integer(as_r_value(keras$ops$convert_to_numpy(x)))
}

## May need to revisit this; either to disable it, or export a custom $<- method
## for base classes like Layer, so that compound assignment expressions aren't a
## problem.
#' @export
py_to_r.keras.src.utils.tracking.TrackedDict <- function(x) import("builtins")$dict(x)

#' @export
py_to_r.keras.src.utils.tracking.TrackedList <- function(x) import("builtins")$list(x)

#' @export
py_to_r.keras.src.utils.tracking.TrackedSet <- function(x) import("builtins")$list(x)
