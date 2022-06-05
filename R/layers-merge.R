

#' Layer that adds a list of inputs.
#'
#' It takes as input a list of tensors, all of the same shape, and returns a
#' single tensor (also of the same shape).
#'
#' @param inputs A input tensor, or list of input tensors. Can be missing.
#' @param ... Unnamed args are treated as additional `inputs`. Named arguments are passed on as standard layer arguments.
#'
#' @return A tensor, the sum of the inputs. If `inputs` is missing, a keras
#'   layer instance is returned.
#'
#' @family merging_layers
#'
#' @seealso
#' +  <https://www.tensorflow.org/api_docs/python/tf/keras/layers/add>
#' +  <https://keras.io/api/layers/merging_layers/add>
#'
#' @export
layer_add <- function(inputs, ...) {
  if (missing(inputs))
    return(keras$layers$Add(...))
  if (!is.list(inputs))
    inputs <- list(inputs)
  dots <- split_dots_named_unnamed(list(...))
  inputs <- c(inputs, dots$unnamed)
  do.call(keras$layers$add, c(list(inputs), dots$named))
}


# TODO: there should be a common topic where we can use
# @inheritDotParams standard-layer-args


#' Layer that subtracts two inputs.
#'
#' It takes as input a list of tensors of size 2, both of the same shape, and
#' returns a single tensor, (`inputs[[1]] - inputs[[2]]`), also of the same
#' shape.
#'
#' @param inputs A input tensor, or list of two input tensors. Can be missing.
#' @param ... Unnamed args are treated as additional `inputs`. Named arguments are passed on as standard layer arguments.
#'
#' @return A tensor, the difference of the inputs. If `inputs` is missing, a
#'   keras layer instance is returned.
#'
#' @family merge layers
#'
#'
#' @seealso
#' +  <https://www.tensorflow.org/api_docs/python/tf/keras/layers/subtract>
#' +  <https://keras.io/api/layers/merging_layers/subtract>
#'
#' @export
layer_subtract <- function(inputs, ...) {
  if (missing(inputs))
    return(keras$layers$Subtract(...))
  if (!is.list(inputs))
    inputs <- list(inputs)
  dots <- split_dots_named_unnamed(list(...))
  inputs <- c(inputs, dots$unnamed)
  do.call(keras$layers$subtract, c(list(inputs), dots$named))
}


#' Layer that multiplies (element-wise) a list of inputs.
#'
#' It takes as input a list of tensors, all of the same shape, and returns a
#' single tensor (also of the same shape).
#'
#' @param inputs A input tensor, or list of input tensors. Can be missing.
#' @param ... Unnamed args are treated as additional `inputs`. Named arguments are passed on as standard layer arguments.
#'
#' @return A tensor, the element-wise product of the inputs. If `inputs` is
#'   missing, a keras layer instance is returned.
#'
#' @family merge layers
#'
#'
#' @seealso
#' +  <https://www.tensorflow.org/api_docs/python/tf/keras/layers/multiply>
#' +  <https://keras.io/api/layers/merging_layers/multiply>
#'
#' @export
layer_multiply <- function(inputs, ...) {
  if (missing(inputs))
    return(keras$layers$Multiply(...))
  if (!is.list(inputs))
    inputs <- list(inputs)
  dots <- split_dots_named_unnamed(list(...))
  inputs <- c(inputs, dots$unnamed)
  do.call(keras$layers$multiply, c(list(inputs), dots$named))
}


#' Layer that averages a list of inputs.
#'
#' It takes as input a list of tensors, all of the same shape, and returns a
#' single tensor (also of the same shape).
#'
#' @param inputs A input tensor, or list of input tensors. Can be missing.
#' @param ... Unnamed args are treated as additional `inputs`. Named arguments are passed on as standard layer arguments.
#'
#' @return A tensor, the average of the inputs. If `inputs` is missing, a keras
#'   layer instance is returned.
#'
#' @family merge layers
#'
#'
#' @seealso
#' +  <https://www.tensorflow.org/api_docs/python/tf/keras/layers/average>
#' +  <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Average>
#' +  <https://keras.io/api/layers/merging_layers/average>
#'
#' @export
layer_average <- function(inputs, ...) {
  if (missing(inputs))
    return(keras$layers$Average(...))
  if (!is.list(inputs))
    inputs <- list(inputs)
  dots <- split_dots_named_unnamed(list(...))
  inputs <- c(inputs, dots$unnamed)
  do.call(keras$layers$average, c(list(inputs), dots$named))
}

#' Layer that computes the maximum (element-wise) a list of inputs.
#'
#' It takes as input a list of tensors, all of the same shape, and returns a
#' single tensor (also of the same shape).
#'
#' @param inputs A input tensor, or list of input tensors. Can be missing.
#' @param ... Unnamed args are treated as additional `inputs`. Named arguments are passed on as standard layer arguments.
#'
#' @return A tensor, the element-wise maximum of the inputs. If `inputs` is
#'   missing, a keras layer instance is returned.
#'
#' @family merge layers
#'
#'
#' @seealso
#' +  <https://www.tensorflow.org/api_docs/python/tf/keras/layers/maximum>
#' +  <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Maximum>
#' +  <https://keras.io/api/layers/merging_layers/maximum>
#'
#' @export
layer_maximum <- function(inputs, ...) {
  if (missing(inputs))
    return(keras$layers$Maximum(...))
  if (!is.list(inputs))
    inputs <- list(inputs)
  dots <- split_dots_named_unnamed(list(...))
  inputs <- c(inputs, dots$unnamed)
  do.call(keras$layers$maximum, c(list(inputs), dots$named))
}


#' Layer that computes the minimum (element-wise) a list of inputs.
#'
#' It takes as input a list of tensors, all of the same shape, and returns a
#' single tensor (also of the same shape).
#'
#' @param inputs A input tensor, or list of input tensors. Can be missing.
#' @param ... Unnamed args are treated as additional `inputs`. Named arguments are passed on as standard layer arguments.
#'
#' @return A tensor, the element-wise maximum of the inputs. If `inputs` is
#'   missing, a keras layer instance is returned.
#'
#' @family merge layers
#'
#' @seealso
#' +  <https://www.tensorflow.org/api_docs/python/tf/keras/layers/minimum>
#' +  <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Minimum>
#' +  <https://keras.io/api/layers/merging_layers/minimum>
#'
#' @export
layer_minimum <- function(inputs, ...) {
  if (missing(inputs))
    return(keras$layers$Minimum(...))
  if (!is.list(inputs))
    inputs <- list(inputs)
  dots <- split_dots_named_unnamed(list(...))
  inputs <- c(inputs, dots$unnamed)
  do.call(keras$layers$minimum, c(list(inputs), dots$named))
}


#' Layer that concatenates a list of inputs.
#'
#' It takes as input a list of tensors, all of the same shape expect for the
#' concatenation axis, and returns a single tensor, the concatenation of all
#' inputs.
#'
#' @param inputs A input tensor, or list of input tensors. Can be missing.
#' @param ... Unnamed args are treated as additional `inputs`. Named arguments are passed on as standard layer arguments.
#' @param axis Concatenation axis.
#'
#' @return A tensor, the concatenation of the inputs alongside axis `axis`. If
#'   `inputs` is missing, a keras layer instance is returned.
#'
#' @family merge layers
#'
#' @seealso
#' +  <https://www.tensorflow.org/api_docs/python/tf/keras/layers/concatenate>
#' +  <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Concatenate>
#' +  <https://keras.io/api/layers/merging_layers/concatenate>
#'
#' @export
layer_concatenate <- function(inputs, ..., axis = -1) {
  if (missing(inputs)) {
    args <- capture_args(match.call(), list(axis = as.integer))
    return(do.call(keras$layers$Concatenate, args))
  }

  # TODO: this axis arg should probably be 1-based

  if (is.list(inputs)) {
    # backcompat: axis used to be in 2nd position, inputs used to accept only a list.

    dots <- list(...)
    if (length(dots) && names2(dots)[[1]] == "" &&
        missing(axis) &&
        is.numeric(dots[[1L]]) &&
        is_scalar(dots[[1L]])) {
      axis <- as.integer(dots[[1L]])
      dots[[1L]] <- NULL
    }
    return(do.call(keras$layers$concatenate,
                   c(list(inputs), dots, axis = as.integer(axis))))
  }

  dots <- split_dots_named_unnamed(list(...))
  inputs <- c(list(inputs), dots$unnamed)
  do.call(keras$layers$concatenate, c(list(inputs), dots$named))
}

#' Layer that computes a dot product between samples in two tensors.
#'
#' @param inputs A input tensor, or list of input tensors. Can be missing.
#' @param ... Unnamed args are treated as additional `inputs`. Named arguments are passed on as standard layer arguments.
#' @param axes Integer or list of integers, axis or axes along which to take the
#'   dot product.
#' @param normalize Whether to L2-normalize samples along the dot product axis
#'   before taking the dot product. If set to TRUE, then the output of the dot
#'   product is the cosine proximity between the two samples.
#'
#' @return If `inputs` is supplied: A tensor, the dot product of the samples
#'   from the inputs. If `inputs` is missing, a keras layer instance is
#'   returned.
#'
#'
#' @family merge layers
#'
#' @seealso
#' +  <https://www.tensorflow.org/api_docs/python/tf/keras/layers/dot>
#' +  <https://keras.io/api/layers/merging_layers/dot/>
#'
#' @export
#' @importFrom rlang names2
layer_dot <- function(inputs, ..., axes, normalize = FALSE) {
  if (missing(inputs)) {
    args <- capture_args(match.call(), list(axes = as.integer))
    return(do.call(keras$layers$Dot, args))
  }

  if (is.list(inputs)) {
    # backcompat: inputs used to only accept a list of layers, and
    # axis, normalize, used to be in 2nd, 3rd position.
    dots <- list(...)
    if (length(dots) && names2(dots)[[1]] == "" &&
        missing(axes)) {
      axes <- as.integer(dots[[1L]])
      dots[[1L]] <- NULL
    }
    if (length(dots) && names2(dots)[[1]] == "" &&
        missing(normalize)) {
      normalize  <- as.integer(dots[[1L]])
      dots[[1L]] <- NULL
    }
    args <- c(list(inputs), dots,
              axes = as.integer(axes), normalize = normalize)
    return(do.call(keras$layers$dot, args))
  }

  # inputs is not a list
  dots <- split_dots_named_unnamed(list(...))
  inputs <- c(inputs, dots$unnamed)
  args <- c(list(inputs),
            dots$named,
            axes = as.integer(axes),
            normalize = normalize)
  do.call(keras$layers$dot, args)
}
