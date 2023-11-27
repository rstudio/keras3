
#' @export
`==.keras.backend.common.keras_tensor.KerasTensor` <- function(e1, e2) {
  k_equal(e1, e2)
}

#' @export
`+.keras.backend.common.keras_tensor.KerasTensor` <- function(e1, e2) {
  if(missing(e2)) return(e1)
  NextMethod()
}
