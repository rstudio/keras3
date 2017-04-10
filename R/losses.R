
#' Model loss functions
#' 
#' @param y_true True labels (TensorFlow tensor) 
#' @param y_pred Predictions (TensorFlow tensor of the same shape as `y_true`)
#' 
#' @details Loss functions are to be supplied in the `loss` parameter of the 
#' [compile()] function.
#' 
#' Loss functions can be specified either using the name of a built in loss
#' function (e.g. 'loss = binary_crossentropy'), a reference to a built in loss
#' function (e.g. 'loss = loss_binary_crossentropy()') or by passing an
#' artitrary function that returns a scalar for each data-point and takes the
#' following two arguments: 
#' 
#' - `y_true` True labels (TensorFlow tensor) 
#' - `y_pred` Predictions (TensorFlow tensor of the same shape as `y_true`)
#' 
#' The actual optimized objective is the mean of the output array across all
#' datapoints.
#' 
#' @section Categorical Crossentropy:
#'   
#'   When using the categorical_crossentropy loss, your targets should be in
#'   categorical format (e.g. if you have 10 classes, the target for each sample
#'   should be a 10-dimensional vector that is all-zeros expect for a 1 at the
#'   index corresponding to the class of the sample). In order to convert
#'   integer targets into categorical targets, you can use the Keras utility
#'   function [to_categorical()]:
#'   
#'   `categorical_labels <- to_categorical(int_labels, num_classes = NULL)`
#'   
#' @seealso [compile()]   
#'   
#' @export
loss_mean_squared_error <- function(y_true, y_pred) {
  if (missing(y_true) && missing(y_pred))
    keras$losses$mean_squared_error
  else
    keras$losses$mean_squared_error(y_true, y_pred)
}

#' @rdname loss_mean_squared_error
#' @export
loss_mean_absolute_error <- function(y_true, y_pred) {
  if (missing(y_true) && missing(y_pred))
    keras$losses$mean_absolute_error
  else
    keras$losses$mean_absolute_error(y_true, y_pred)
}

#' @rdname loss_mean_squared_error
#' @export
loss_mean_absolute_percentage_error <- function(y_true, y_pred) {
  if (missing(y_true) && missing(y_pred))
    keras$losses$mean_absolute_percentage_error
  else
    keras$losses$mean_absolute_percentage_error(y_true, y_pred)
}

#' @rdname loss_mean_squared_error
#' @export
loss_mean_squared_logarithmic_error <- function(y_true, y_pred) {
  if (missing(y_true) && missing(y_pred))
    keras$losses$mean_squared_logarithmic_error
  else
    keras$losses$mean_squared_logarithmic_error(y_true, y_pred)
}

#' @rdname loss_mean_squared_error
#' @export
loss_squared_hinge <- function(y_true, y_pred) {
  if (missing(y_true) && missing(y_pred))
    keras$losses$squared_hinge
  else
    keras$losses$squared_hinge(y_true, y_pred)
}

#' @rdname loss_mean_squared_error
#' @export
loss_hinge <- function(y_true, y_pred) {
  if (missing(y_true) && missing(y_pred))
    keras$losses$hinge
  else
    keras$losses$hinge(y_true, y_pred)
}

#' @rdname loss_mean_squared_error
#' @export
loss_categorical_crossentropy <- function(y_true, y_pred) {
  if (missing(y_true) && missing(y_pred))
    keras$losses$categorical_crossentropy
  else
    keras$losses$categorical_crossentropy(y_true, y_pred)
}


#' @rdname loss_mean_squared_error
#' @export
loss_sparse_categorical_crossentropy <- function(y_true, y_pred) {
  if (missing(y_true) && missing(y_pred))
    keras$losses$sparse_categorical_crossentropy
  else
    keras$losses$sparse_categorical_crossentropy(y_true, y_pred)
}

#' @rdname loss_mean_squared_error
#' @export
loss_binary_crossentropy <- function(y_true, y_pred) {
  if (missing(y_true) && missing(y_pred))
    keras$losses$binary_crossentropy
  else
    keras$losses$binary_crossentropy(y_true, y_pred)
}

#' @rdname loss_mean_squared_error
#' @export
loss_kullback_leibler_divergence <- function(y_true, y_pred) {
  if (missing(y_true) && missing(y_pred))
    keras$losses$kullback_leibler_divergence
  else
    keras$losses$kullback_leibler_divergence(y_true, y_pred)
}

#' @rdname loss_mean_squared_error
#' @export
loss_poisson <- function(y_true, y_pred) {
  if (missing(y_true) && missing(y_pred))
    keras$losses$poisson
  else
    keras$losses$poisson(y_true, y_pred)
}

#' @rdname loss_mean_squared_error
#' @export
loss_cosine_proximity <- function(y_true, y_pred) {
  if (missing(y_true) && missing(y_pred))
    keras$losses$cosine_proximity
  else
    keras$losses$cosine_proximity(y_true, y_pred)
}





