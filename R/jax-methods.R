
#' @export
as.array.jaxlib.xla_extension.ArrayImpl <- function(x, ...) {
  import("numpy")$asarray(x)
}

#' @export
as.double.jaxlib.xla_extension.ArrayImpl <- function(x, ...) {
  as.double(import("numpy")$asarray(x))
}
