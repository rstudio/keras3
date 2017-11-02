

#' Layer that adds a list of inputs.
#' 
#' It takes as input a list of tensors, all of the same shape, and returns a
#' single tensor (also of the same shape).
#' 
#' @inheritParams layer_dense
#' 
#' @param inputs A list of input tensors (at least 2).
#'   
#' @return A tensor, the sum of the inputs.
#'   
#' @family merge layers
#'   
#' @export
layer_add <- function(inputs, batch_size = NULL, dtype = NULL, 
                      name = NULL, trainable = NULL, weights = NULL) {
  keras$layers$add(
    inputs = inputs,
    batch_size = as_nullable_integer(batch_size),
    dtype = dtype,
    name = name,
    trainable = trainable,
    weights = weights
  )
}


#' Layer that subtracts two inputs.
#'
#' It takes as input a list of tensors of size 2, both of the same shape, and
#' returns a single tensor, (`inputs[[1]] - inputs[[2]]``), also of the same
#' shape.
#'
#' @inheritParams layer_dense
#'
#' @param inputs A list of input tensors (exactly 2).
#'
#' @return A tensor, the difference of the inputs.
#'
#' @family merge layers
#'
#' @export
layer_subtract <- function(inputs, batch_size = NULL, dtype = NULL, 
                           name = NULL, trainable = NULL, weights = NULL) {
  keras$layers$subtract(
    inputs = inputs,
    batch_size = as_nullable_integer(batch_size),
    dtype = dtype,
    name = name,
    trainable = trainable,
    weights = weights
  )
}

#' Layer that multiplies (element-wise) a list of inputs.
#' 
#' It takes as input a list of tensors, all of the same shape, and returns a
#' single tensor (also of the same shape).
#'   
#' @inheritParams layer_dense
#'   
#' @param inputs A list of input tensors (at least 2).
#'   
#' @return A tensor, the element-wise product of the inputs.
#' 
#' @family merge layers
#' 
#' @export
layer_multiply <- function(inputs, batch_size = NULL, dtype = NULL, 
                           name = NULL, trainable = NULL, weights = NULL) {
  keras$layers$multiply(
    inputs = inputs,
    batch_size = as_nullable_integer(batch_size),
    dtype = dtype,
    name = name,
    trainable = trainable,
    weights = weights
  )
}


#' Layer that averages a list of inputs.
#' 
#' It takes as input a list of tensors, all of the same shape, and returns a
#' single tensor (also of the same shape).
#' 
#' @inheritParams layer_dense
#' 
#' @param inputs A list of input tensors (at least 2).
#'   
#' @return A tensor, the average of the inputs.
#'   
#' @family merge layers
#'   
#' @export
layer_average <- function(inputs, batch_size = NULL, dtype = NULL, 
                          name = NULL, trainable = NULL, weights = NULL) {
  keras$layers$average(
    inputs = inputs,
    batch_size = as_nullable_integer(batch_size),
    dtype = dtype,
    name = name,
    trainable = trainable,
    weights = weights
  )
}

#' Layer that computes the maximum (element-wise) a list of inputs.
#' 
#' It takes as input a list of tensors, all of the same shape, and returns a
#' single tensor (also of the same shape).
#' 
#' @inheritParams layer_dense
#' 
#' @param inputs A list of input tensors (at least 2). 
#'   
#' @return A tensor, the element-wise maximum of the inputs.
#'  
#' @family merge layers   
#'     
#' @export
layer_maximum <- function(inputs, batch_size = NULL, dtype = NULL, 
                          name = NULL, trainable = NULL, weights = NULL) {
  keras$layers$maximum(
    inputs = inputs,
    batch_size = as_nullable_integer(batch_size),
    dtype = dtype,
    name = name,
    trainable = trainable,
    weights = weights
  )
}


#' Layer that computes the minimum (element-wise) a list of inputs.
#' 
#' It takes as input a list of tensors, all of the same shape, and returns a
#' single tensor (also of the same shape).
#' 
#' @inheritParams layer_dense
#' 
#' @param inputs A list of input tensors (at least 2). 
#'   
#' @return A tensor, the element-wise maximum of the inputs.
#'  
#' @family merge layers   
#'     
#' @export
layer_minimum <- function(inputs, batch_size = NULL, dtype = NULL, 
                          name = NULL, trainable = NULL, weights = NULL) {
  keras$layers$minimum(
    inputs = inputs,
    batch_size = as_nullable_integer(batch_size),
    dtype = dtype,
    name = name,
    trainable = trainable,
    weights = weights
  )
}


#' Layer that concatenates a list of inputs.
#' 
#' It takes as input a list of tensors, all of the same shape expect for the
#' concatenation axis, and returns a single tensor, the concatenation of all
#' inputs.
#'   
#' @inheritParams layer_dense
#'   
#' @param inputs A list of input tensors (at least 2).
#' @param axis Concatenation axis.
#'   
#' @return A tensor, the concatenation of the inputs alongside axis `axis`.
#'   
#' @family merge layers   
#'   
#' @export
layer_concatenate <- function(inputs, axis = -1, batch_size = NULL, dtype = NULL, 
                              name = NULL, trainable = NULL, weights = NULL) {
  keras$layers$concatenate(
    inputs = inputs,
    axis = as.integer(axis),
    batch_size = as_nullable_integer(batch_size),
    dtype = dtype,
    name = name,
    trainable = trainable,
    weights = weights
  )
}

#' Layer that computes a dot product between samples in two tensors.
#' 
#' @inheritParams layer_dense
#' 
#' @param inputs A list of input tensors (at least 2).
#' @param axes Integer or list of integers, axis or axes along which to take the dot product.
#' @param normalize Whether to L2-normalize samples along the dot product axis before taking the dot product. If set to TRUE, then the output of the dot product is the cosine proximity between the two samples. **kwargs: Standard layer keyword arguments.
#' 
#' @return A tensor, the dot product of the samples from the inputs.
#' 
#' @family merge layers
#' 
#' @export
layer_dot <- function(inputs, axes, normalize = FALSE, batch_size = NULL, dtype = NULL, 
                      name = NULL, trainable = NULL, weights = NULL) {
  keras$layers$dot(
    inputs = inputs,
    axes = as.integer(axes),
    normalize = normalize,
    batch_size = as_nullable_integer(batch_size),
    dtype = dtype,
    name = name,
    trainable = trainable,
    weights = weights
  )
}




