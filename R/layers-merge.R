

#' Layer that adds a list of inputs.
#'
#' It takes as input a list of tensors, all of the same shape, and returns a
#' single tensor (also of the same shape).
#'
#' @param inputs A list of input tensors (at least 2). Can be missing.
#' @param ... Standard layer arguments (must be named).
#'
#' @return A tensor, the sum of the inputs. If `inputs` is missing, a keras
#'   layer instance is returned.
#'
#' @family merging_layers
#'
#' @seealso
#' +  <https://www.tensorflow.org/api_docs/python/tf/keras/layers/add>
#' +  <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Add>
#' +  <https://keras.io/api/layers/merging_layers/add>
#'
#' @export
layer_add <- function(inputs, ...) {
  callable <- if(missing(inputs)) keras$layers$Add else keras$layers$add
  args <- capture_args(match.call(), list(batch_size = as_nullable_integer))
  do.call(callable, args)
}

# TODO: there should be a common topic where we can use
# @inheritDotParams standard-layer-args


#' Layer that subtracts two inputs.
#'
#' It takes as input a list of tensors of size 2, both of the same shape, and
#' returns a single tensor, (`inputs[[1]] - inputs[[2]]`), also of the same
#' shape.
#'
#' @param inputs A list of input tensors (exactly 2). Can be missing.
#' @param ... Standard layer arguments (must be named).
#'
#' @return A tensor, the difference of the inputs. If `inputs` is missing, a
#'   keras layer instance is returned.
#'
#' @family merge layers
#'
#'
#' @seealso
#' +  <https://www.tensorflow.org/api_docs/python/tf/keras/layers/subtract>
#' +  <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Subtract>
#' +  <https://keras.io/api/layers/merging_layers/subtract>
#'
#' @export
layer_subtract <- function(inputs, ...) {
  callable <- if (missing(inputs)) keras$layers$Subtract else keras$layers$subtract
  args <- capture_args(match.call(), list(batch_size = as_nullable_integer))
  do.call(callable, args)
}

#' Layer that multiplies (element-wise) a list of inputs.
#'
#' It takes as input a list of tensors, all of the same shape, and returns a
#' single tensor (also of the same shape).
#'
#' @param inputs A list of input tensors (at least 2). Can be missing.
#' @param ... Standard layer arguments (must be named).
#'
#' @return A tensor, the element-wise product of the inputs. If `inputs` is
#'   missing, a keras layer instance is returned.
#'
#' @family merge layers
#'
#'
#' @seealso
#' +  <https://www.tensorflow.org/api_docs/python/tf/keras/layers/multiply>
#' +  <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Multiply>
#' +  <https://keras.io/api/layers/merging_layers/multiply>
#'
#' @export
layer_multiply <- function(inputs, ...) {
  callable <- if (missing(inputs)) keras$layers$Multiply else keras$layers$multiply
  args <- capture_args(match.call(), list(batch_size = as_nullable_integer))
  do.call(callable, args)

}


#' Layer that averages a list of inputs.
#'
#' It takes as input a list of tensors, all of the same shape, and returns a
#' single tensor (also of the same shape).
#'
#' @param inputs A list of input tensors (at least 2). Can be missing.
#' @param ... Standard layer arguments (must be named).
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
  callable <- if (missing(inputs)) keras$layers$Average else keras$layers$average
  args <- capture_args(match.call(), list(batch_size = as_nullable_integer))
  do.call(callable, args)

}

#' Layer that computes the maximum (element-wise) a list of inputs.
#'
#' It takes as input a list of tensors, all of the same shape, and returns a
#' single tensor (also of the same shape).
#'
#' @param inputs A list of input tensors (at least 2). Can be missing.
#' @param ... Standard layer arguments (must be named).
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
  callable <- if (missing(inputs)) keras$layers$Maximum else keras$layers$maximum
  args <- capture_args(match.call(), list(batch_size = as_nullable_integer))
  do.call(callable, args)

}


#' Layer that computes the minimum (element-wise) a list of inputs.
#'
#' It takes as input a list of tensors, all of the same shape, and returns a
#' single tensor (also of the same shape).
#'
#' @param inputs A list of input tensors (at least 2). Can be missing.
#' @param ... Standard layer arguments (must be named).
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
  callable <- if (missing(inputs)) keras$layers$Minimum else keras$layers$minimum
  args <- capture_args(match.call(), list(batch_size = as_nullable_integer))
  do.call(callable, args)
}


#' Layer that concatenates a list of inputs.
#'
#' It takes as input a list of tensors, all of the same shape expect for the
#' concatenation axis, and returns a single tensor, the concatenation of all
#' inputs.
#'
#' @param inputs A list of input tensors (at least 2). Can be missing.
#' @param axis Concatenation axis.
#' @param ... Standard layer arguments (must be named).
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
layer_concatenate <- function(inputs, axis = -1, ...) {
  callable <- if (missing(inputs)) keras$layers$Concatenate else keras$layers$concatenate
  # TODO: this axis should probably be 1-based
  args <- capture_args(match.call(), list(batch_size = as_nullable_integer,
                                          axis = as.integer))
  do.call(callable, args)
}

#' Layer that computes a dot product between samples in two tensors.
#'
#' @param inputs A list of input tensors (at least 2). Can be missing.
#' @param axes Integer or list of integers, axis or axes along which to take the
#'   dot product.
#' @param normalize Whether to L2-normalize samples along the dot product axis
#'   before taking the dot product. If set to TRUE, then the output of the dot
#'   product is the cosine proximity between the two samples.
#' @param ... Standard layer arguments (must be named).
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
#' +  <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dot>
#' +  <https://keras.io/api/layers/merging_layers/dot/>
#'
#' @export
layer_dot <- function(inputs, axes, normalize = FALSE, ...) {
  callable <- if (missing(inputs)) keras$layers$Dot else keras$layers$dot
  args <- capture_args(match.call(), list(batch_size = as_nullable_integer,
                                          axes = as.integer))
  do.call(callable, args)
}
