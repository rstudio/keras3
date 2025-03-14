

try_into_slice <- function(x) {
  if(is.integer(x) && length(x) >= 2L) {
    rng <- range(x)
    start <- rng[1L]
    end <- rng[2L]
    step <-  x[2L] - x[1L]

    x2 <- do.call(seq.int, as.list(c(start, end, by = step)))
    if (identical(x, x2)) {
      if (identical(step, 1L))
        step <- NULL
      if (identical(start, 1L))
        start <- NULL
      return(asNamespace("reticulate")$py_slice(start, end, step))
    }
  }
  NULL
}

broadcast_to_rank <- function(x, axis, rank) {
  if (is.atomic(x)) {
    shp <- rep_len(1L, rank)
    shp[axis] <- length(x)
    dim(x) <- shp
  } else {
    if (rank > 1L)
      x <- op_expand_dims(x, seq_len(rank)[-axis])
  }
  x
}


op_subset <- function(x, ...) {
  key <- asNamespace("reticulate")$dots_to__getitem__key(..., .envir = parent.frame())
  key <- if (inherits(key, "python.builtin.tuple"))
    py_to_r(key)
  else
    list(key)

  delayedAssign("x_shape", op_shape(x))
  delayedAssign("x_rank", op_ndim(x))


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
      return(py_get_item(x, as.array(arg)))
    }

    if(is.matrix(arg) && ncol(arg) == x_rank) {
      storage.mode(arg) <- "integer"
      pos <- arg > 0L
      arg[pos] <- arg[pos] - 1L
      key <- tuple(asplit(arg, 2))
      return(py_get_item(x, key))
    }

    if (op_is_tensor(arg)) {
      if (op_dtype(arg) == "bool") {
        return(py_get_item(x, arg))
      }

      if (op_ndim(arg) == 2L && op_shape(arg)[[2]] == x_rank) {
        arg <- op_where(arg > 0, arg - 1L, arg)
        key <- tuple(op_unstack(arg, axis = 2))
        return(py_get_item(x, key))
      }
    }
  }


  advanced_idx <- logical(...length())

  translate_one_based_to_zero_based <- function(arg, axis) {

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
      if (!is.null(end)) {
        if (is.numeric(end)) {
          end <- if (identical(end, -1L) || identical(end, x_shape[[axis]]))
            NULL
          else
            as.integer(end) + 1L
        } else if (op_is_tensor(end)) {
          end <- op_where(end > 0L, end, (x_shape[[axis]] + 1L) - end)
        }
      }
      slice2 <- asNamespace("reticulate")$py_slice(start, end, arg$step)
      return(slice2)

    }

    if (inherits(arg, "python.builtin.ellipsis")) {
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
        # TODO: should op_nonzero cbind the results?
        arg <- op_nonzero(arg)[[1L]]
      } else {
        arg <- op_where(arg > 0L, arg - 1L, arg)
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


  key <- imap(key, translate_one_based_to_zero_based)
  if (any(advanced_idx)) {
    advanced_subspace_rank <- sum(advanced_idx)
    key[advanced_idx] <- imap(key[advanced_idx], function(arg, subspace_axis) {
      broadcast_to_rank(arg, subspace_axis, advanced_subspace_rank)
    })
  }

  key <- tuple(key)
  # print(key)
  py_get_item(x, key)
}

