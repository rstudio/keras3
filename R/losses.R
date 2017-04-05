
#' Model loss functions
#'   
#' @note 
#'   Loss functions are to be supplied in the `loss` parameter of the 
#'   [compile()] function. 
#' 
#'   Loss functions can be specified either using the name of a built 
#'   in loss function (e.g. 'loss = binary_crossentropy'), a reference to
#'   a built in loss function (e.g. 'loss = loss_binary_crossentropy()') or 
#'   by passing an artitrary function that returns a scalar for each data-point
#'   and takes the following two arguments:
#'   - `y_true` True labels (TensorFlow tensor)
#'   - `y_pred` Predictions (TensorFlow tensor of the same shape as `y_true`)
#'
#'   The actual optimized objective is the mean of the output array across all datapoints.
#'   
#' @seealso [compile()]   
#'   
#' @export
loss_mean_squared_error <- function() {
  keras$losses$mean_squared_error
}

#' @rdname loss_mean_squared_error
#' @export
loss_mean_absolute_error <- function() {
  keras$losses$mean_absolute_error
}

#' @rdname loss_mean_squared_error
#' @export
loss_mean_absolute_percentage_error <- function() {
  keras$losses$mean_absolute_percentage_error
}

#' @rdname loss_mean_squared_error
#' @export
loss_mean_squared_logarithmic_error <- function() {
  keras$losses$mean_squared_logarithmic_error
}

#' @rdname loss_mean_squared_error
#' @export
loss_squared_hinge <- function() {
  keras$losses$squared_hinge
}

#' @rdname loss_mean_squared_error
#' @export
loss_hinge <- function() {
  keras$losses$hinge
}

#' @rdname loss_mean_squared_error
#' @export
loss_categorical_crossentropy <- function() {
  keras$losses$categorical_crossentropy
}


#' @rdname loss_mean_squared_error
#' @export
loss_sparse_categorical_crossentropy <- function() {
  keras$losses$sparse_categorical_crossentropy
}

#' @rdname loss_mean_squared_error
#' @export
loss_binary_crossentropy <- function() {
  keras$losses$binary_crossentropy
}

#' @rdname loss_mean_squared_error
#' @export
loss_kullback_leibler_divergence <- function() {
  keras$losses$kullback_leibler_divergence
}

#' @rdname loss_mean_squared_error
#' @export
loss_poisson <- function() {
  keras$losses$poisson
}

#' @rdname loss_mean_squared_error
#' @export
loss_cosine_proximity <- function() {
  keras$losses$cosine_proximity
}





