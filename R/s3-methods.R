
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

#' @exportS3Method base::all.equal
all.equal.keras.src.backend.common.variables.KerasVariable <-
function(target, current, ...) {
  if (inherits(target, "keras.src.backend.common.variables.KerasVariable"))
    target <- as_r_value(target$numpy())
  if (inherits(current, "keras.src.backend.common.variables.KerasVariable"))
    current <- as_r_value(current$numpy())
  all.equal(target, current, ...)
}


## This method isn't the best semantic match for all.equal(), but identical()
## isn't a generic, and doesn't work correctly for comparing python objects (it
## returns false if the pyref environment isn't the same exact environment, even
## if the pyrefs are wrapping the same py object), and there isn't a great
## (exported) way to compare if two # tensors are the same that doesn't leak
## python concepts...
#' @exportS3Method base::all.equal
all.equal.keras.src.backend.common.keras_tensor.KerasTensor <-
function(target, current, ...) {
  inherits(target, "keras.src.backend.common.keras_tensor.KerasTensor") &&
  inherits(current, "keras.src.backend.common.keras_tensor.KerasTensor") &&
  py_id(target) == py_id(current)
}

