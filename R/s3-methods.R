
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
  as_r_value(ops$convert_to_numpy(x))
}
#' @export
as.array.keras.src.backend.Variable <- as.array.keras.src.backend.common.variables.KerasVariable


#' @export
as.numeric.keras.src.backend.common.variables.KerasVariable <- function(x, ...) {
  as.numeric(as_r_value(ops$convert_to_numpy(x)))
}
#' @export
as.numeric.keras.src.backend.Variable <- as.numeric.keras.src.backend.common.variables.KerasVariable

#' @export
as.double.keras.src.backend.common.variables.KerasVariable <- function(x, ...) {
  as.double(as_r_value(ops$convert_to_numpy(x)))
}
#' @export
as.double.keras.src.backend.Variable <- as.double.keras.src.backend.common.variables.KerasVariable

#' @export
as.integer.keras.src.backend.common.variables.KerasVariable <- function(x, ...) {
  as.integer(as_r_value(ops$convert_to_numpy(x)))
}
#' @export
as.integer.keras.src.backend.Variable <- as.integer.keras.src.backend.common.variables.KerasVariable


#' @exportS3Method base::all.equal
all.equal.keras.src.backend.common.variables.KerasVariable <-
function(target, current, ...) {
  if (inherits(target, "keras.src.backend.common.variables.KerasVariable"))
    target <- as_r_value(target$numpy())
  if (inherits(current, "keras.src.backend.common.variables.KerasVariable"))
    current <- as_r_value(current$numpy())
  all.equal(target, current, ...)
}
all.equal.keras.src.backend.Variable <- all.equal.keras.src.backend.common.variables.KerasVariable


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
#' @exportS3Method base::all.equal
all.equal.keras.src.backend.Tensor <- all.equal.keras.src.backend.common.keras_tensor.KerasTensor


## Conditionally export these py_to_r methods, if tensorflow hasn't already exported them.
## We do this to keep keras3 and tensorflow decoupled, but to avoid
## "S3 method overwritten" warnings if both packages are loaded.
##
## Note, we still may need to revisit this; either to disable it, or export a custom $<- method
## for base classes like Layer, so that compound assignment expressions aren't a
## problem.
##
# these S3 methods are conditionally registered in .onLoad() instead of in NAMESPACE.
# __ instead of . to avoid a roxygen warning about unexported S3 methods when generating NAMESPACE
py_to_r__keras.src.utils.tracking.TrackedDict <- function(x) import("builtins")$dict(x)

py_to_r__keras.src.utils.tracking.TrackedList <- function(x) import("builtins")$list(x)

py_to_r__keras.src.utils.tracking.TrackedSet <- function(x) import("builtins")$list(x)

#  @rawNamespace S3method(as.array,   keras.src.backend.Variable)
#  @rawNamespace S3method(as.numeric, keras.src.backend.Variable)
#  @rawNamespace S3method(as.double,  keras.src.backend.Variable)
#  @rawNamespace S3method(as.integer, keras.src.backend.Variable)
#  @rawNamespace S3method(all.equal,  keras.src.backend.Variable)
#  @rawNamespace S3method(`+`,        keras.src.backend.Variable)
#  @rawNamespace S3method(`==`,       keras.src.backend.Variable)
# for(generic in c("==", "+", "as.array", "as.numeric", "as.double", "as.integer", "all.equal")) {
#   for (cls in c("keras.src.backend.Variable", "keras.src.backend.Tensor"))
#     assign(
#       sprintf("%s.%s", generic, cls),
#       get(sprintf("%s.keras.src.backend.common.keras_tensor.KerasTensor", generic)))
# }
# rm(list = c("generic", "cls"))


