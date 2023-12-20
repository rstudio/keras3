
#' @export
`==.keras.backend.common.keras_tensor.KerasTensor` <- function(e1, e2) {
  op_equal(e1, e2)
}

#' @export
`+.keras.backend.common.keras_tensor.KerasTensor` <- function(e1, e2) {
  if(missing(e2)) return(e1)
  NextMethod()
}


#' @export
as.array.keras.backend.common.variables.KerasVariable <- function(x, ...) {
  as_r_value(keras$ops$convert_to_numpy(x))
}

#' @export
as.numeric.keras.backend.common.variables.KerasVariable <- function(x, ...) {
  as.numeric(as_r_value(keras$ops$convert_to_numpy(x)))
}

#' @export
as.double.keras.backend.common.variables.KerasVariable <- function(x, ...) {
  as.double(as_r_value(keras$ops$convert_to_numpy(x)))
}

#' @export
as.integer.keras.backend.common.variables.KerasVariable <- function(x, ...) {
  as.integer(as_r_value(keras$ops$convert_to_numpy(x)))
}
