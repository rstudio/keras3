


#' Converts a class vector (integers) to binary class matrix.
#' 
#' @details 
#' E.g. for use with [loss_categorical_crossentropy()].
#' 
#' @param y Class vector to be converted into a matrix (integers from 0 to num_classes).
#' @param num_classes Total number of classes.
#' 
#' @return A binary matrix representation of the input.
#' 
#' @export
to_categorical <- function(y, num_classes = NULL) {
  keras$utils$to_categorical(
    y = y,
    num_classes = as_nullable_integer(num_classes)
  )
}

 


is_keras_function <- function(f) {
  is.function(f) && identical(environment(f), getNamespace("keras"))
}

resolve_keras_function <- function(f) {
  if (is_keras_function(f))
    f()
  else
    f
}