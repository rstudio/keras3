
#' Metrics used for judging the performance of models
#' 
#' @param y_true True labels (tensor)
#' @param y_pred Predictions (tensor of the same shape as `y_true`)
#' @param k Top-k
#'   
#' @return Single tensor value representing the mean of the output array across
#'   all datapoints.
#'   
#' @note Metric functions are to be supplied in the `metrics` parameter of the 
#'   [compile()] function.
#'   
#' @section Custom Metrics:
#'   
#'   Custom metrics can be passed at the compilation step. The function would 
#'   need to take `(y_true, y_pred)` as arguments and return a single tensor 
#'   value.
#'   
#' @name model-metrics
#' @export
metric_binary_accuracy <- function(y_true, y_pred) {
  keras$metrics$binary_accuracy(
    y_true = y_true,
    y_pred = y_pred
  )
}

#' @rdname model-metrics
#' @export
metric_binary_crossentropy <- function(y_true, y_pred) {
  keras$metrics$binary_crossentropy(
    y_true = y_true,
    y_pred = y_pred
  )
}

#' @rdname model-metrics
#' @export
metric_categorical_accuracy <- function(y_true, y_pred) {
  keras$metrics$categorical_accuracy(
    y_true = y_true,
    y_pred = y_pred
  )
}

#' @rdname model-metrics
#' @export
metric_categorical_crossentropy <- function(y_true, y_pred) {
  keras$metrics$categorical_crossentropy(
    y_true = y_true,
    y_pred = y_pred
  )
}

#' @rdname model-metrics
#' @export
metric_cosine_proximity <- function(y_true, y_pred) {
  keras$metrics$cosine_proximity(
    y_true = y_true,
    y_pred = y_pred
  )
}

#' @rdname model-metrics
#' @export
metric_hinge <- function(y_true, y_pred) {
  keras$metrics$hinge(
    y_true = y_true,
    y_pred = y_pred
  )
}

#' @rdname model-metrics
#' @export
metric_kullback_leibler_divergence <- function(y_true, y_pred) {
  keras$metrics$kullback_leibler_divergence(
    y_true = y_true,
    y_pred = y_pred
  )
}

#' @rdname model-metrics
#' @export
metric_mean_absolute_error <- function(y_true, y_pred) {
  keras$metrics$mean_absolute_error(
    y_true = y_true,
    y_pred = y_pred
  )
}

#' @rdname model-metrics
#' @export
metric_mean_absolute_percentage_error <- function(y_true, y_pred) {
  keras$metrics$mean_absolute_percentage_error(
    y_true = y_true,
    y_pred = y_pred
  )
}

#' @rdname model-metrics
#' @export
metric_mean_squared_error <- function(y_true, y_pred) {
  keras$metrics$mean_squared_error(
    y_true = y_true,
    y_pred = y_pred
  )
}


#' @rdname model-metrics
#' @export
metric_mean_squared_logarithmic_error <- function(y_true, y_pred) {
  keras$metrics$mean_squared_logarithmic_error(
    y_true = y_true,
    y_pred = y_pred
  )
}


#' @rdname model-metrics
#' @export
metric_poisson <- function(y_true, y_pred) {
  keras$metrics$poisson(
    y_true = y_true,
    y_pred = y_pred
  )
}

#' @rdname model-metrics
#' @export
metric_sparse_categorical_crossentropy <- function(y_true, y_pred) {
  keras$metrics$sparse_categorical_crossentropy(
    y_true = y_true,
    y_pred = y_pred
  )
}

#' @rdname model-metrics
#' @export
metric_squared_hinge <- function(y_true, y_pred) {
  keras$metrics$squared_hinge(
    y_true = y_true,
    y_pred = y_pred
  )
}

#' @rdname model-metrics
#' @export
metric_top_k_categorical_accuracy <- function(y_true, y_pred, k = 5) {
  keras$metrics$top_k_categorical_accuracy(
    y_true = y_true,
    y_pred = y_pred,
    k = as.integer(k)
  )
}















