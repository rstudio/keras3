
#' @export
as.array.jax.Array <- function(x, ...) {
  import("numpy")$asarray(x)
}

#' @export
as.array.jaxlib._jax.ArrayImpl <- as.array.jax.Array

#' @export
as.array.jaxlib.xla_extension.ArrayImpl <- as.array.jax.Array

#' @export
as.double.jax.Array <- function(x, ...) {
  as.double(import("numpy")$asarray(x))
}

#' @export
as.double.jaxlib._jax.ArrayImpl <- as.double.jax.Array

#' @export
as.double.jaxlib.xla_extension.ArrayImpl <- as.double.jax.Array
