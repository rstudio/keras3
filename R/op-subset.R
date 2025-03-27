

try_into_slice <- function(x) {
  if (is.numeric(x) && length(x) >= 2L) {
    x <- as.integer(x)
    start <- x[1L]
    end <- x[length(x)]
    step <-  x[2L] - x[1L]
    if (sign(start) != sign(end))
      return()
    x2 <- seq.int(start, end, step)

    if (identical(as.integer(x), x2)) {
      if (identical(step, 1L))
        step <- NULL
      if (identical(start, 1L))
        start <- NULL
      if(identical(end, -1L))
        end <- NULL
      return(asNamespace("reticulate")$py_slice(start, end, step))
    }
  }
  NULL
}

seq_len0 <- function(x) seq.int(from = 0L, length.out = x)

broadcast_to_rank <- function(x, axis, rank) {
  if (is.atomic(x)) {
    shp <- rep_len(1L, rank)
    shp[axis] <- length(x)
    dim(x) <- shp
  } else if (inherits(x, "tensorflow.tensor")) {
    if (rank > 1L)
      x <- tf$expand_dims(x, seq_len0(rank)[-axis])
  } else if (op_is_tensor(x)) {
    if (rank > 1L)
      x <- ops$expand_dims(x, seq_len0(rank)[-axis])
  } else {
    # stop()?
  }
  x
}


r_extract_args_into_py_get_item_key <- function(x, ..., .envir = parent.frame(2L)) {
  key <- asNamespace("reticulate")$dots_to__getitem__key(..., .envir = .envir)
  key <- if (inherits(key, "python.builtin.tuple"))
    py_to_r(key)
  else
    list(key)

  if(inherits(x, "tensorflow.tensor")) {
    delayedAssign("x_shape", as_r_value(x$shape$as_list()))
    delayedAssign("x_rank", as_r_value(x$ndim))
  } else {
    delayedAssign("x_shape", op_shape(x))
    delayedAssign("x_rank", op_ndim(x))
  }

  # there are these possible values
  # only 1 arg in ..., it is a logical array of shape x
  # - pass it through
  # only 1 arg in ..., it is a matrix with ncol(x) == rank(x)
  # - split along cols, pass as tuple
  #
  # Otherwise, we are subsetting with 1d arrays.
  # in which case:
  #   pass through slices
  #   convert logical to integer


  if (length(key) == 1) {
    arg <- key[[1]]
    if(inherits(arg, "numpy.ndarray"))
      arg <- py_to_r(arg)
    if(is.logical(arg)) {
      return(as.array(arg))
    }

    if(is.matrix(arg) && ncol(arg) == x_rank) {
      storage.mode(arg) <- "integer"
      pos <- arg > 0L
      arg[pos] <- arg[pos] - 1L
      key <- tuple(asplit(arg, 2))
      return(key)
    }

    if (inherits(arg, "numpy.ndarray")) {
      if (as_r_value(x$dtype$name) == "bool") {
        return(arg)
      }
      if (as_r_value(arg$ndim) == 2L &&
          as_r_value(arg$shape)[[2L]] == x_rank) {

        arg <- as_py_index(arg)
        # arg <- np$where(arg > 0L, arg - 1L, arg)
        key <- np$unstack(arg, axis = 1L)
        return(key)
      }
    }

    # need a separate check for tensorflow tensors, since op_is_tensor()
    # only detects the configured backend, but we might still get a tf.tensor
    if(inherits(arg, "tensorflow.tensor")) {
      if (as_r_value(arg$dtype$is_bool)) {
        return(arg)
      }
      if (as_r_value(arg$ndim) == 2L &&
          as_r_value(arg$shape$as_list())[[2L]] == x_rank) {

        arg <- tf$where(arg > 0L, arg - 1L, arg)
        key <- tuple(tf$unstack(arg, axis = 1L))
        return(key)
      }
    }

    if (op_is_tensor(arg)) {
      if (op_dtype(arg) == "bool") {
        return(arg)
      }

      if (op_ndim(arg) == 2L && op_shape(arg)[[2]] == x_rank) {
        arg <- ops$where(arg > 0L, arg - 1L, arg)
        key <- tuple(op_unstack(arg, axis = 2L))
        return(key)
      }
    }
  }


  advanced_idx <- logical(...length())

  into_py_index <- function(arg, axis) {

    if (is.null(arg)) {
      return() # newaxis
    }

    # TODO might want to handle this in Python for bigint support
    if (inherits(arg, c("numpy.ndarray",
                        "python.builtin.float",
                        "python.builtin.int" )))
      arg <- py_to_r(arg)


    if (is.logical(arg)) {
      arg <- as.array(arg)
      if (length(dim(arg)) != 1L) {
        stop("logical must be rank 1 if subsetting a single axis")
      }

      arg <- as.array(which(arg) - 1L)
      advanced_idx[axis] <<- TRUE
      return(arg)
    }


    if (is.double(arg))
      storage.mode(arg) <- "integer"

    if (is.integer(arg)) {
      if (length(arg) == 1L) {
        if (arg == 0L)
          stop("indexing is 1 based")
        if (arg > 0L)
          arg <- arg - 1L
        return(arg)
      }

      if (length(arg) > 1L) {
        if (!is.null(slice <- try_into_slice(arg))) {
          slice <- sys.function()(slice, axis)
          return(slice)
        }

        arg <- as.array(arg)
        if (length(dim(arg)) > 1L)
          stop("subsetting with integer vectors of rank > 1 not implemented")
        pos <- arg > 0
        arg[pos] <- arg[pos] - 1L
        advanced_idx[axis] <<- TRUE
        return(arg)

      } else {
        stop("subsetting with length 0 integer vectors not supported")
      }
    }


    if (inherits(arg, "python.builtin.slice")) {
      start <- sys.function()(py_to_r(arg$start), axis)
      if(identical(start, 0L))
        start <- NULL
      end <- py_to_r(arg$stop)
      step <- py_to_r(arg$step) %||% 1L
      if (!is.null(end)) {
        if (is.integer(end)) {
          if (identical(end, x_shape[[axis]])) {
            end <- NULL
          } else if (end < 0L) {
            end2 <- end + step # TODO: handle case where step is a tensor
            if (sign(end2) != sign(end))
              end <- NULL
            else
              end <- end2
          }
        } else if(inherits(end, "tensorflow.tensor")) {
          end <- tf$where(end > 0L, end, (x_shape[[axis]] + 1L) - end)
        } else if (op_is_tensor(end)) {
          end <- ops$where(end > 0L, end, (x_shape[[axis]] + 1L) - end)
        }
      }
      if (identical(step, 1L))
        step <- NULL
      slice2 <- asNamespace("reticulate")$py_slice(start, end, step)
      return(slice2)

    }

    if (inherits(arg, "python.builtin.ellipsis")) {
      return(arg)
    }

    if (inherits(arg, "tensorflow.tensor")) {
      arg_rank <- as_r_value(arg$ndim)
      if (arg_rank == 0L) {
        # scalar
        arg <- tf$where(arg > 0L, arg - 1L, arg)
        return(arg)
      }

      if (arg_rank != 1L) {
        stop("subsetting tensor must be 1d")
      }

      if (as_r_value(arg$dtype$is_bool)) {
        arg <- tf$squeeze(tf$where(arg), 1L)
      } else {
        arg <- tf$where(arg > 0L, arg - 1L, arg)
      }

      advanced_idx[axis] <<- TRUE
      return(arg)
    }

    # a tensor (eager or symbolic).
    if (op_is_tensor(arg)) {
      arg_rank <- op_ndim(arg)
      if (arg_rank == 0) {
        # scalar
        arg <- op_where(arg > 0L, arg - 1L, arg)
        return(arg)
      }

      if (arg_rank != 1L) {
        stop("subsetting tensor must be 1d")
      }

      if (op_dtype(arg) == "bool") {
        arg <- ops$nonzero(arg)[[1L]]
      } else {
        arg <- ops$where(arg > 0L, arg - 1L, arg)
      }

      advanced_idx[axis] <<- TRUE
      return(arg)
    }

    # warning("Unrecognized object")
    # str(arg)
    # print(class(arg))
    # browser()
    # stop("Unrecognized object")
    arg

  }


  key <- imap(key, into_py_index)
  if (any(advanced_idx)) {
    advanced_subspace_rank <- sum(advanced_idx)
    key[advanced_idx] <- imap(key[advanced_idx], function(arg, subspace_axis) {
      broadcast_to_rank(arg, subspace_axis, advanced_subspace_rank)
    })
  }

  # TODO: add support for drop
  # e.g., if(drop) {key <- lapply(key, function(x) {
  #   if(is_scalar(x)) py_slice(x, x) else x
  #   # also, check if 'drop' needed for indices
  # })

  key <- tuple(key)
  key
}


#' Subset elements from a tensor
#'
#' Extract elements from a tensor using common R-style `[` indexing idioms. This
#' function can also be conveniently accessed via the syntax `tensor@r[...]`.
#'
#' @param x Input tensor.
#'
#' @param ... Indices specifying elements to extract. Each argument in `...` can
#'   be:
#'
#' - An integer scalar
#' - A 1-d integer or logical vector
#' - `NULL` or `newaxis`
#' - The `..` symbol
#' - A slice expression using `:`
#'
#' If only a single argument is supplied to `...`, then `..1` can also be:
#'
#' - A logical array with the same shape as `x`
#' - An integer matrix where `ncol(..1) == op_rank(x)`
#'
#' @param value new value to replace the selected subset with.
#'
#' @details
#'
#' While the semantics are similar to R's `[`, there are some differences:
#'
#' # Differences from R's `[`:
#'
#' - Negative indices follow Python-style indexing, counting from the end of the array.
#' - `NULL` or `newaxis` adds a new dimension (equivalent to `op_expand_dims()`).
#' - If fewer indices than dimensions (`op_rank(x)`) are provided, missing dimensions
#'   are implicitly filled. For example, if `x` is a matrix, `x[1]` returns the first row.
#' - `..` or `all_dims()` expands to include all unspecified dimensions (see examples).
#' - Extended slicing syntax (`:`) is supported, including:
#'   - Strided steps: `x@r[start:end:step]`
#'   - `NA` values for `start` and `end`. `NA` for `start` defaults to `1`, and
#'     `NA` for `end` defaults to the axis size.
#' - A logical array matching the shape of `x` selects elements in row-wise order.
#'
#' # Similarities with R's `[`:
#'
#' Similarities to R's `[` (differences from Python's `[`):
#'
#' - Positive indices are 1-based.
#' - Slices (`x[start:end]`) are inclusive of `end`.
#' - 1-d logical/integer arrays subset along their respective axis.
#'   Multiple vectors provided for different axes return intersected subsets.
#' - A single integer matrix with `ncol(i) == op_rank(x)` selects elements by
#'   coordinates. Each row in the matrix specifies the location of one value, where each column
#'   corresponds to an axis in the tensor being subsetted.  This means you use a
#'   2-column matrix to subset a matrix, a 3-column matrix to subset a 3d array,
#'   and so on.
#'
#' # Examples
#'
#' ```{r}
#' (x <- op_arange(5L) + 10L)
#'
#' # Basic example, get first element
#' op_subset(x, 1)
#'
#' # Use `@r[` syntax
#' x@r[1]           # same as `op_subset(x, 1)`
#' x@r[1:2]         # get the first 2 elements
#' x@r[c(1, 3)]     # first and third element
#'
#' # Negative indices
#' x@r[-1]          # last element
#' x@r[-2]          # second to last element
#' x@r[c(-1, -2)]   # last and second to last elements
#' x@r[c(-2, -1)]   # second to last and last elements
#' x@r[c(1, -1)]    # first and last elements
#'
#' # Slices
#' x@r[1:3]          # first 3 elements
#' x@r[NA:3]         # first 3 elements
#' x@r[1:5]          # all elements
#' x@r[1:-1]         # all elements
#' x@r[NA:NA]        # all elements
#' x@r[]             # all elements
#'
#' x@r[1:-2]         # drop last element
#' x@r[NA:-2]        # drop last element
#' x@r[2:NA]         # drop first element
#'
#' # 2D array examples
#' xr <- array(1:12, c(3, 4))
#' x <- op_convert_to_tensor(xr)
#'
#' # Basic subsetting
#' x@r[1, ]      # first row
#' x@r[1]        # also first row! Missing axes are implicitly inserted
#' x@r[-1]       # last row
#' x@r[, 2]      # second column
#' x@r[, 2:2]    # second column, but shape preserved (like [, drop=FALSE])
#'
#' # Subsetting with a boolean array
#' # Note: extracted elements are selected row-wise, not column-wise
#' mask <- x >= 6
#' x@r[mask]             # returns a 1D tensor
#'
#' x.r <- as.array(x)
#' mask.r <- as.array(mask)
#' # as.array(x)[mask] selects column-wise. Use `aperm()` to reverse search order.
#' all(aperm(x.r)[aperm(mask.r)] == as.array(x@r[mask]))
#'
#' # Subsetting with a matrix of index positions
#' indices <- rbind(c(1, 1), c(2, 2), c(3, 3))
#' x@r[indices] # get diagonal elements
#' x.r[indices] # same as subsetting an R array
#'
#'
#' # 3D array examples
#' # Image: 4x4 pixels, 3 colors (RGB)
#' # Tensor shape: (img_height, img_width, img_color_channels)
#' shp <- shape(4, 4, 3)
#' x <- op_arange(prod(shp)) |> op_reshape(shp)
#'
#' # Convert to a batch of images by inserting a new axis
#' # New shape: (batch_size, img_height, img_width, img_color_channels)
#' x@r[newaxis, , , ] |> op_shape()
#' x@r[newaxis] |> op_shape()  # same as above
#' x@r[NULL] |> op_shape()     # same as above
#'
#' x <- x@r[newaxis]
#' # Extract color channels
#' x@r[, , , 1]          # red channel
#' x@r[.., 1]            # red channel, same as above using .. shorthand
#' x@r[.., 2]            # green channel
#' x@r[.., 3]            # blue channel
#'
#' # .. expands to all unspecified axes.
#' op_shape(x@r[])
#' op_shape(x@r[..])
#' op_shape(x@r[1, ..])
#' op_shape(x@r[1, .., 1, 1])
#' op_shape(x@r[1, 1, 1, .., 1])
#'
#'
#' # op_subset<- uses the same semantics, but note that not all tensors
#' # support modification. E.g., TensorFlow constant tensors cannot be modified,
#' # while TensorFlow Variables can be.
#'
#' (x <- tensorflow::tf$Variable(matrix(1, nrow = 2, ncol = 3)))
#' op_subset(x, 1) <- 9
#' x
#'
#' x@r[1,1] <- 33
#' x
#' ```
#'
#' @returns
#' A tensor containing the subset of elements.
#'
#' @export
#' @family core ops
#' @family ops
op_subset <- function(x, ...) {
  key <- r_extract_args_into_py_get_item_key(x, ..., .envir = parent.frame())
  # print(key)
  if (inherits(x, "tensorflow.tensor")) {
    return(tf_numpy_style_get_item(x, key))
  }
  py_get_item(x, key)
}

tf_numpy_style_get_item <- function(x, key, silent = FALSE) {
  if (inherits(x, "tensorflow.tensor")) {
    numpy_style_getitem <- py_get_attr(x, "_numpy_style_getitem", TRUE)
    if (!is.null(numpy_style_getitem)) {
      return(numpy_style_getitem(key))
    }
  }
  py_get_item(x, key, silent = silent)
}


#' @export
#' @rdname op_subset
#' @family core ops
#' @family ops
`op_subset<-` <- function(x, ..., value) {
  # browser()
  key <- r_extract_args_into_py_get_item_key(x, ..., .envir = parent.frame())
  if (is.atomic(value)) {
    if (is.double(x) && grepl("int", op_dtype(x)))
      storage.mode(value) <- "integer"
    if (length(value) > 1L)
      value <- as.array(value)
  }

  repeat { # paired with `break` for a simulacrum of a goto

    # handle jax and tensorflow, regardless of backend
    if (inherits(x, "tensorflow.tensor")) {
      # x[0, 0].assign(3.)
      x_subset <- tf_numpy_style_get_item(x, key, TRUE) %||% break
      assign <- py_get_attr(x_subset, "assign", TRUE) %||% break
      # assign returns a new ref to the same variable, with
      # the assignment op as a parent op in graph mode.
      return(assign(value))
    }

    if (any(startsWith(class(x), "jax"))) {
      # "jaxlib.xla_extension.ArrayImpl", other S3 classes?
      # new_x = x.at[0].set(10)
      x_at <- py_get_attr(x, "at", TRUE) %||% break
      x_subset <- py_get_item(x_at, key, TRUE) %||% break
      set <- py_get_attr(x, "set") %||% break
      return(set(value))
    }
    break
  }

  # default method; torch, numpy, ...
  # will throw error if object does not support in-place modification
  py_set_item(x, key, value)

}


#' @export
#' @rdname op_subset
#' @family core ops
#' @family ops
op_subset_set <- `op_subset<-`
