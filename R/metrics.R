
#' Model performance metrics
#'   
#' @note   
#' Metric functions are to be supplied in the `metrics` parameter of the 
#'   [compile()] function.
#'   
#' @export
metric_binary_accuracy <- function() {
  keras$metrics$binary_accuracy
}

#' @rdname metric_binary_accuracy
#' @export
metric_binary_crossentropy <- function() {
  keras$metrics$binary_crossentropy
}

#' @rdname metric_binary_accuracy
#' @export
metric_categorical_accuracy <- function() {
  keras$metrics$categorical_accuracy
}

#' @rdname metric_binary_accuracy
#' @export
metric_categorical_crossentropy <- function() {
  keras$metrics$categorical_crossentropy
}

#' @rdname metric_binary_accuracy
#' @export
metric_cosine_proximity <- function() {
  keras$metrics$cosine_proximity
}

#' @rdname metric_binary_accuracy
#' @export
metric_hinge <- function() {
  keras$metrics$hinge
}

#' @rdname metric_binary_accuracy
#' @export
metric_kullback_leibler_divergence <- function() {
  keras$metrics$kullback_leibler_divergence
}

#' @rdname metric_binary_accuracy
#' @export
metric_mean_absolute_error <- function() {
  keras$metrics$mean_absolute_error
}

#' @rdname metric_binary_accuracy
#' @export
metric_mean_absolute_percentage_error <- function() {
  keras$metrics$mean_absolute_percentage_error
}

#' @rdname metric_binary_accuracy
#' @export
metric_mean_squared_error <- function() {
  keras$metrics$mean_squared_error
}


#' @rdname metric_binary_accuracy
#' @export
metric_mean_squared_logarithmic_error <- function() {
  keras$metrics$mean_squared_logarithmic_error
}


#' @rdname metric_binary_accuracy
#' @export
metric_poisson <- function() {
  keras$metrics$poisson
}

#' @rdname metric_binary_accuracy
#' @export
metric_sparse_categorical_crossentropy <- function() {
  keras$metrics$sparse_categorical_crossentropy
}

#' @rdname metric_binary_accuracy
#' @export
metric_squared_hinge <- function() {
  keras$metrics$squared_hinge
}

#' @rdname metric_binary_accuracy
#' @export
metric_top_k_categorical_accuracy <- function() {
  keras$metrics$top_k_categorical_accuracy
}















