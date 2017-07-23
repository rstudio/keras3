
#' Model performance metrics
#'   
#' @note   
#' Metric functions are to be supplied in the `metrics` parameter of the 
#'   [compile()] function.
#' 
#' @param y_true True labels (tensor)
#' @param y_pred Predictions (tensor of the same shape as y_true).
#' @param k An integer, number of top elements to consider.
#'       
#' @section Custom Metrics:
#' You can provide an arbitrary R function as a custom metric. Note that
#' the `y_true` and `y_pred` parameters are tensors, so computations on 
#' them should use backend tensor functions. For example:
#' 
#' ```r
#' # create metric using backend tensor functions
#' K <- backend()
#' metric_mean_pred <- function(y_true, y_pred) {
#'   K$mean(y_pred) 
#' }
#' 
#' model %>% compile( 
#'   optimizer = optimizer_rmsprop(),
#'   loss = loss_binary_crossentropy,
#'   metrics = c('accuracy', 
#'               'mean_pred' = metric_mean_pred)
#' )
#' ```
#' 
#' Note that a name ('mean_pred') is provided for the custom metric
#' function. This name is used within training progress output.
#' 
#' Documentation on the available backend tensor functions can be 
#' found at <https://rstudio.github.io/keras/articles/backend.html#backend-functions>.     
#'
#' @export
metric_binary_accuracy <- function(y_true, y_pred) {
  keras$metrics$binary_accuracy(y_true, y_pred)
}
attr(metric_binary_accuracy, "py_function_name") <- "binary_accuracy"

#' @rdname metric_binary_accuracy
#' @export
metric_binary_crossentropy <- function(y_true, y_pred) {
  keras$metrics$binary_crossentropy(y_true, y_pred)
}
attr(metric_binary_crossentropy, "py_function_name") <- "binary_crossentropy"


#' @rdname metric_binary_accuracy
#' @export
metric_categorical_accuracy <- function(y_true, y_pred) {
  keras$metrics$categorical_accuracy(y_true, y_pred)
}
attr(metric_categorical_accuracy, "py_function_name") <- "categorical_accuracy"


#' @rdname metric_binary_accuracy
#' @export
metric_categorical_crossentropy <- function(y_true, y_pred) {
  keras$metrics$categorical_crossentropy(y_true, y_pred)
}
attr(metric_categorical_crossentropy, "py_function_name") <- "categorical_crossentropy"


#' @rdname metric_binary_accuracy
#' @export
metric_cosine_proximity <- function(y_true, y_pred) {
  keras$metrics$cosine_proximity(y_true, y_pred)
}
attr(metric_cosine_proximity, "py_function_name") <- "cosine_proximity"


#' @rdname metric_binary_accuracy
#' @export
metric_hinge <- function(y_true, y_pred) {
  keras$metrics$hinge(y_true, y_pred)
}
attr(metric_hinge, "py_function_name") <- "hinge"


#' @rdname metric_binary_accuracy
#' @export
metric_kullback_leibler_divergence <- function(y_true, y_pred) {
  keras$metrics$kullback_leibler_divergence(y_true, y_pred)
}
attr(metric_kullback_leibler_divergence, "py_function_name") <- "kullback_leibler_divergence"


#' @rdname metric_binary_accuracy
#' @export
metric_mean_absolute_error <- function(y_true, y_pred) {
  keras$metrics$mean_absolute_error(y_true, y_pred)
}
attr(metric_mean_absolute_error, "py_function_name") <- "mean_absolute_error"



#' @rdname metric_binary_accuracy
#' @export
metric_mean_absolute_percentage_error <- function(y_true, y_pred) {
  keras$metrics$mean_absolute_percentage_error(y_true, y_pred)
}
attr(metric_mean_absolute_percentage_error, "py_function_name") <- "mean_absolute_percentage_error"


#' @rdname metric_binary_accuracy
#' @export
metric_mean_squared_error <- function(y_true, y_pred) {
  keras$metrics$mean_squared_error(y_true, y_pred)
}
attr(metric_mean_squared_error, "py_function_name") <- "mean_squared_error"



#' @rdname metric_binary_accuracy
#' @export
metric_mean_squared_logarithmic_error <- function(y_true, y_pred) {
  keras$metrics$mean_squared_logarithmic_error(y_true, y_pred)
}
attr(metric_mean_squared_logarithmic_error, "py_function_name") <- "mean_squared_logarithmic_error"


#' @rdname metric_binary_accuracy
#' @export
metric_poisson <- function(y_true, y_pred) {
  keras$metrics$poisson(y_true, y_pred)
}
attr(metric_poisson, "py_function_name") <- "poisson"


#' @rdname metric_binary_accuracy
#' @export
metric_sparse_categorical_crossentropy <- function(y_true, y_pred) {
  keras$metrics$sparse_categorical_crossentropy(y_true, y_pred)
}
attr(metric_sparse_categorical_crossentropy, "py_function_name") <- "sparse_categorical_crossentropy"



#' @rdname metric_binary_accuracy
#' @export
metric_squared_hinge <- function(y_true, y_pred) {
  keras$metrics$squared_hinge(y_true, y_pred)
}
attr(metric_squared_hinge, "py_function_name") <- "squared_hinge"



#' @rdname metric_binary_accuracy
#' @export
metric_top_k_categorical_accuracy <- function(y_true, y_pred, k = 5) {
  keras$metrics$top_k_categorical_accuracy(y_true, y_pred, k = as.integer(k))
}
attr(metric_top_k_categorical_accuracy, "py_function_name") <- "top_k_categorical_accuracy"



#' @rdname metric_binary_accuracy
#' @export
metric_sparse_top_k_categorical_accuracy <- function(y_true, y_pred, k = 5) {
  keras$metrics$sparse_top_k_categorical_accuracy(y_true, y_pred, k = as.integer(k))
}
attr(metric_sparse_top_k_categorical_accuracy, "py_function_name") <- "sparse_top_k_categorical_accuracy"













