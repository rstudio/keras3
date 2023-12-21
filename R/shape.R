
#' Tensor shape utility
#'
#' This function can be used to create or get the shape of an object.
#'
#' # Examples
#' ```{r}
#' shape(1, 2, 3)
#' ```
#'
#' 3 ways to specify an unknown dimension
#' ```{r, results = "hold"}
#' shape(NA,   2, 3)
#' shape(NULL, 2, 3)
#' shape(-1,   2, 3)
#' ```
#'
#' Most functions that take a 'shape' argument also coerce with `shape()`
#' ```{r, results = "hold"}
#' layer_input(c(1, 2, 3))
#' layer_input(shape(1, 2, 3))
#' ```
#'
#' You can also use `shape()` to get the shape of a tensor
#' ```{r}
#' symbolic_tensor <- layer_input(shape(1, 2, 3))
#' shape(symbolic_tensor)
#'
#' eager_tensor <- op_ones(c(1,2,3))
#' shape(eager_tensor)
#' op_shape(eager_tensor)
#' ```
#'
#' Combine or expand shapes
#' ```{r}
#' shape(symbolic_tensor, 4)
#' shape(5, symbolic_tensor, 4)
#' ```
#'
#' In graph mode, a shape might contain a scalar integer tensor for unknown
#' axes.
#' ```{r}
#' tfn <- tensorflow::tf_function(function(x) {
#'   print(shape(x))
#'   x
#' },
#' input_signature = list(tensorflow::tf$TensorSpec(shape(1, NA, 3))))
#' invisible(tfn(op_ones(shape(1, 2, 3))))
#' ```
#'
#' A useful pattern is to unpack the `shape()` with `%<-%`, like this:
#' ```r
#' c(batch_size, seq_len, channels) %<-% shape(x)
#' ```
#'
#' If you are unpacking `shape()` in graph mode, and then want to reassemble the
#' axes with `shape()`, you'll have to wrap tensors with `I()` to use the tensor
#' itself, rather than the shape of the tensor.
#' ```{r}
#' echo_print <- function(x) { message("> ", deparse(substitute(x))); print(x) }
#' tfn <- tensorflow::tf_function(function(x) {
#'   c(axis1, axis2, axis3) %<-% shape(x)
#'   str(list(axis1 = axis1, axis2 = axis2, axis3 = axis3))
#'
#'   echo_print(shape(axis2))               # resolve axis2 tensor shape
#'   echo_print(shape(axis1, axis2, axis3)) # resolve axis2 tensor shape
#'
#'   echo_print(shape(I(axis2)))               # use axis2 tensor as axis value
#'   echo_print(shape(axis1, I(axis2), axis3)) # use axis2 tensor as axis value
#'   x
#' },
#' input_signature = list(tensorflow::tf$TensorSpec(shape(1, NA, 3))))
#' invisible(tfn(op_ones(shape(1, 2, 3))))
#' ```
#'
#' @param ... A shape specification. Numerics, `NULL` and tensors are valid.
#'   `NULL`, `NA`, and `-1L` can be used to specify an unspecified dim size.
#'   Tensors are dispatched to `k_shape()` to extract the tensor shape. Values
#'   wrapped in `I()` are used asis (see examples). All other objects are coerced
#'   via `as.integer()`.
#'
#' @return A list with a `"keras_shape"` class attribute. Each element of the
#'   list will be either a) `NULL`, b) an integer or c) a scalar integer tensor
#'   (e.g., when supplied a TF tensor with a unspecified dimension in a function
#'   being traced).
#'
#'
#' @export
#' @seealso [op_shape()]
shape <- function(...) {

  fix <- function(x) {
    if (inherits(x, "AsIs")) {
      class(x) <- setdiff(class(x), "AsIs")
      return(x)
    }

    if (inherits(x, 'python.builtin.object')) {
      if (inherits(x, "tensorflow.python.framework.tensor_shape.TensorShape"))
        return(as.integer(x))

      tryCatch({
        return(lapply(keras$ops$shape(x),
                      function(d) as_r_value(d) %||% NA_integer_))
      }, error = identity)
    }

    if (!is.atomic(x) || length(x) > 1)
      lapply(x, fix)
    else if (is.null(x) ||
             identical(x, NA_integer_) ||
             identical(x, NA_real_) ||
             identical(x, NA) ||
             (is.numeric(x) && isTRUE(suppressWarnings(x == -1L))))
      NA_integer_ # so we can safely unlist()
    else
      as.integer(x)
  }

  shp <- unlist(fix(list(...)), use.names = FALSE)
  shp <- lapply(shp, function(x) if (identical(x, NA_integer_)) NULL else x)
  class(shp) <- "keras_shape"
  shp
}

#' @export
#' @rdname shape
format.keras_shape <- function(x, ...) {
  x <- vapply(x, function(d) format(d %||% "NA"), "")
  x <- paste0(x, collapse = ", ")
  paste0("shape(", x, ")")
}

#' @export
#' @rdname shape
print.keras_shape <- function(x, ...) {
  writeLines(format(x, ...))
  invisible(x)
}

#' @rdname shape
#' @export
`[.keras_shape` <- function(x, ...) {
  out <- unclass(x)[...]
  class(out) <- class(x)
  out
}

#' @rdname shape
#' @export
r_to_py.keras_shape <- function(x, convert = FALSE) {
  tuple(x)
}

#' @rdname shape
#' @export
as.integer.keras_shape <- function(x, ...) {
  vapply(x, function(el) el %||% NA_integer_, 1L)
}

#' @importFrom zeallot destructure
#' @export
destructure.keras_shape <- function(x) unclass(x)

#' @rdname shape
#' @export
as.list.keras_shape <- function(x, ...) unclass(x)
