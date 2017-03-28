

is_keras_function <- function(f) {
  is.function(f) && identical(environment(f), getNamespace("keras"))
}

resolve_keras_function <- function(f) {
  if (is_keras_function(f))
    f()
  else
    f
}