
#' Tensor shape utility
#'
#' This function can be used to get or create a tensor shape.
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
#' (excepting scalar integer tensors).
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
#' Scalar integer tensors are treated as axis values. These are most commonly
#' encountered when tracing a function in graph mode, where an axis size might
#' be unknown.
#' ```{r}
#' tfn <- tensorflow::tf_function(function(x) {
#'   print(op_shape(x))
#'   x
#' },
#' input_signature = list(tensorflow::tf$TensorSpec(shape(1, NA, 3))))
#' invisible(tfn(op_ones(shape(1, 2, 3))))
#' ```
#'
#' A useful pattern is to unpack the `shape()` with `%<-%`, like this:
#' ```r
#' c(batch_size, seq_len, channels) %<-% shape(x)
#'
#' # `%<-%` also has support for skipping values
#' # during unpacking with `.` and `...`. For example,
#' # To retrieve just the first and/or last dim:
#' c(batch_size, ...) %<-% shape(x)
#' c(batch_size, ., .) %<-% shape(x)
#' c(..., channels) %<-% shape(x)
#' c(batch_size, ..., channels) %<-% shape(x)
#' c(batch_size, ., channels) %<-% shape(x)
#' ```
#'
#' ```{r}
#' echo_print <- function(x) {
#'   message("> ", deparse(substitute(x)));
#'   if(!is.null(x)) print(x)
#' }
#' tfn <- tensorflow::tf_function(function(x) {
#'   c(axis1, axis2, axis3) %<-% shape(x)
#'   echo_print(str(list(axis1 = axis1, axis2 = axis2, axis3 = axis3)))
#'
#'   echo_print(shape(axis1))               # use axis1 tensor as axis value
#'   echo_print(shape(axis1, axis2, axis3)) # use axis1 tensor as axis value
#'
#'   # use shape() to compose a new shape, e.g., in multihead attention
#'   n_heads <- 4
#'   echo_print(shape(axis1, axis2, n_heads, axis3/n_heads))
#'
#'   x
#' },
#' input_signature = list(tensorflow::tf$TensorSpec(shape(NA, 4, 16))))
#' invisible(tfn(op_ones(shape(2, 4, 16))))
#' ```
#'
#' If you want to resolve the shape of a tensor that can potentially be
#' a scalar integer, you can wrap the tensor in `I()`, or use [`op_shape()`].
#' ```{r}
#' (x <- op_convert_to_tensor(2L))
#'
#' # by default, shape() treats scalar integer tensors as axis values
#' shape(x)
#'
#' # to access the shape of a scalar integer,
#' # call `op_shape()`, or protect with `I()`
#' op_shape(x)
#' shape(I(x))
#' ```
#'
#' @param ... A shape specification. Numerics, `NULL` and tensors are valid.
#'   `NULL`, `NA`, and `-1L` can be used to specify an unspecified dim size.
#'   Tensors are dispatched to `op_shape()` to extract the tensor shape. Values
#'   wrapped in `I()` are used asis (see examples). All other objects are coerced
#'   via `as.integer()`.
#'
#' @returns A list with a `"keras_shape"` class attribute. Each element of the
#'   list will be either a) `NULL`, b) an R integer or c) a scalar integer tensor
#'   (e.g., when supplied a TF tensor with an unspecified dimension in a function
#'   being traced).
#'
#' @export
#' @seealso [op_shape()]
shape <- function(...) {

  fix <- function(x) {

    if (is_py_object(x)) {
      if (inherits(x, "tensorflow.python.framework.tensor_shape.TensorShape"))
        return(map_int(as.list(as_r_value(x$as_list())),
                       function(e) e %||% NA_integer_))

      shp <- keras$ops$shape(x)

      # convert subclassed tuples, as encountered in Torch
      # class(shp): torch.Size, python.builtin.tuple, python.builtin.object
      if(inherits(shp, "python.builtin.tuple"))
        shp <- import("builtins")$tuple(shp)

      # scalar integer tensors, unprotected with I(), are treated as an axis value
      if (identical(shp, list()) && keras$backend$is_int_dtype(x$dtype)) {
        if (!inherits(x, "AsIs"))
          return(x)
      }

      # otherwise, (most common path) shape() is a tensor shape accessor
      return(lapply(shp, function(d) d %||% NA_integer_))
    }

    if(is.array(x))
      return(dim(x))

    if (is.null(x) ||
        identical(x, NA_integer_) ||
        identical(x, NA_real_) ||
        identical(x, NA) ||
        (is.numeric(x) && isTRUE(suppressWarnings(x == -1L))))
      return(NA_integer_) # so we can safely unlist()

    if (!is.atomic(x) || length(x) > 1)
      return(lapply(x, fix)) # recurse

    as.integer(x)
  }

  shp <- unlist(lapply(list(...), fix), use.names = FALSE)
  shp <- lapply(shp, function(x) if (identical(x, NA_integer_)) NULL else x)
  class(shp) <- "keras_shape"
  shp
}

#' @export
#' @rdname shape
#' @param x A `keras_shape` object.
#' @param prefix Whether to format the shape object with a prefix. Defaults to
#'   `"shape"`.
format.keras_shape <- function(x, ..., prefix = TRUE) {
  x <- vapply(x, function(d) format(d %||% "NA"), "")
  x <- paste0(x, collapse = ", ")
  if(isTRUE(prefix))
    prefix <- "shape"
  else if (!is_string(prefix))
    prefix <-  ""
  paste0(prefix, "(", x, ")")
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

#' @export
r_to_py.keras_shape <- function(x, convert = FALSE) {
  tuple(x, convert = convert)
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

#' @rdname shape
#' @export
`==.keras_shape` <- function(e1, e2) {
  if(!inherits(e1, "keras_shape"))
    e1 <- shape(e1)
  if(!inherits(e2, "keras_shape"))
    e2 <- shape(e2)
  identical(e1, e2)
}

#' @rdname shape
#' @export
`!=.keras_shape` <- function(e1, e2) {
  !`==.keras_shape`(e1, e2)
}


# ' @rdname shape
# ' @export
# c.keras_shape <- function(...) shape(...)
