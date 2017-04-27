
#' Model performance metrics
#'   
#' @note   
#' Metric functions are to be supplied in the `metrics` parameter of the 
#'   [compile()] function.
#' 
#' @param y_true True labels (tensor)
#' @param y_pred Predictions (tensor of the same shape as y_true).
#'       
#' @export
metric_binary_accuracy <- function(y_true, y_pred) {
  if (missing(y_true) && missing(y_pred))
    keras$metrics$binary_accuracy
  else
    keras$metrics$binary_accuracy(y_true, y_pred)
}

#' @rdname metric_binary_accuracy
#' @export
metric_binary_crossentropy <- function(y_true, y_pred) {
  if (missing(y_true) && missing(y_pred))
    keras$metrics$binary_crossentropy
  else
    keras$metrics$binary_crossentropy(y_true, y_pred)
}

#' @rdname metric_binary_accuracy
#' @export
metric_categorical_accuracy <- function(y_true, y_pred) {
  if (missing(y_true) && missing(y_pred))
    keras$metrics$categorical_accuracy
  else
    keras$metrics$categorical_accuracy(y_true, y_pred)
}

#' @rdname metric_binary_accuracy
#' @export
metric_categorical_crossentropy <- function(y_true, y_pred) {
  if (missing(y_true) && missing(y_pred))
    keras$metrics$categorical_crossentropy
  else
    keras$metrics$categorical_crossentropy(y_true, y_pred)
}

#' @rdname metric_binary_accuracy
#' @export
metric_cosine_proximity <- function(y_true, y_pred) {
  if (missing(y_true) && missing(y_pred))
    keras$metrics$cosine_proximity
  else
    keras$metrics$cosine_proximity(y_true, y_pred)
}

#' @rdname metric_binary_accuracy
#' @export
metric_hinge <- function(y_true, y_pred) {
  if (missing(y_true) && missing(y_pred))
    keras$metrics$hinge
  else
    keras$metrics$hinge(y_true, y_pred)
}

#' @rdname metric_binary_accuracy
#' @export
metric_kullback_leibler_divergence <- function(y_true, y_pred) {
  if (missing(y_true) && missing(y_pred))
    keras$metrics$kullback_leibler_divergence
  else
    keras$metrics$kullback_leibler_divergence(y_true, y_pred)
}

#' @rdname metric_binary_accuracy
#' @export
metric_mean_absolute_error <- function(y_true, y_pred) {
  if (missing(y_true) && missing(y_pred))
    keras$metrics$mean_absolute_error
  else
    keras$metrics$mean_absolute_error(y_true, y_pred)
}

#' @rdname metric_binary_accuracy
#' @export
metric_mean_absolute_percentage_error <- function(y_true, y_pred) {
  if (missing(y_true) && missing(y_pred))
    keras$metrics$mean_absolute_percentage_error
  else
    keras$metrics$mean_absolute_percentage_error(y_true, y_pred)
}

#' @rdname metric_binary_accuracy
#' @export
metric_mean_squared_error <- function(y_true, y_pred) {
  if (missing(y_true) && missing(y_pred))
    keras$metrics$mean_squared_error
  else
    keras$metrics$mean_squared_error(y_true, y_pred)
}


#' @rdname metric_binary_accuracy
#' @export
metric_mean_squared_logarithmic_error <- function(y_true, y_pred) {
  if (missing(y_true) && missing(y_pred))
    keras$metrics$mean_squared_logarithmic_error
  else
    keras$metrics$mean_squared_logarithmic_error(y_true, y_pred)
}


#' @rdname metric_binary_accuracy
#' @export
metric_poisson <- function(y_true, y_pred) {
  if (missing(y_true) && missing(y_pred))
    keras$metrics$poisson
  else
    keras$metrics$poisson(y_true, y_pred)
}

#' @rdname metric_binary_accuracy
#' @export
metric_sparse_categorical_crossentropy <- function(y_true, y_pred) {
  if (missing(y_true) && missing(y_pred))
    keras$metrics$sparse_categorical_crossentropy
  else
    keras$metrics$sparse_categorical_crossentropy(y_true, y_pred)
}

#' @rdname metric_binary_accuracy
#' @export
metric_squared_hinge <- function(y_true, y_pred) {
  if (missing(y_true) && missing(y_pred))
    keras$metrics$squared_hinge
  else
    keras$metrics$squared_hinge(y_true, y_pred)
}

#' @rdname metric_binary_accuracy
#' @export
metric_top_k_categorical_accuracy <- function(y_true, y_pred) {
  if (missing(y_true) && missing(y_pred))
    keras$metrics$top_k_categorical_accuracy
  else
    keras$metrics$top_k_categorical_accuracy(y_true, y_pred)
}















