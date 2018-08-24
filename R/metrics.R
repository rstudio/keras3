
#' Model performance metrics
#'   
#' @note   
#' Metric functions are to be supplied in the `metrics` parameter of the 
#'   [compile()] function.
#' 
#' @param y_true True labels (tensor)
#' @param y_pred Predictions (tensor of the same shape as y_true).
#' @param k An integer, number of top elements to consider.
#' @param name Name of custom metric
#' @param metric_fn Custom metric function
#'       
#' @section Custom Metrics:
#' You can provide an arbitrary R function as a custom metric. Note that
#' the `y_true` and `y_pred` parameters are tensors, so computations on 
#' them should use backend tensor functions.
#' 
#' Use the `custom_metric()` function to define a custom metric.
#' Note that a name ('mean_pred') is provided for the custom metric
#' function: this name is used within training progress output.
#' See below for an example.
#' 
#' If you want to save and load a model with custom metrics, you should
#' also specify the metric in the call the [load_model_hdf5()]. For example:
#' `load_model_hdf5("my_model.h5", c('mean_pred' = metric_mean_pred))`. 
#' 
#' Alternatively, you can wrap all of your code in a call to 
#' [with_custom_object_scope()] which will allow you to refer to the 
#' metric by name just like you do with built in keras metrics.
#' 
#' Documentation on the available backend tensor functions can be 
#' found at <https://keras.rstudio.com/articles/backend.html#backend-functions>.
#' 
#' @section Metrics with Parameters:
#' 
#' To use metrics with parameters (e.g. `metric_top_k_categorical_accurary()`)
#' you should create a custom metric that wraps the call with the parameter.
#' See below for an example.
#' 
#' @examples \dontrun{
#' 
#' # create metric using backend tensor functions
#' metric_mean_pred <- custom_metric("mean_pred", function(y_true, y_pred) {
#'   k_mean(y_pred) 
#' })
#' 
#' model %>% compile(
#'   optimizer = optimizer_rmsprop(),
#'   loss = loss_binary_crossentropy,
#'   metrics = c('accuracy', metric_mean_pred)
#' )
#' 
#' # create custom metric to wrap metric with parameter
#' metric_top_3_categorical_accuracy <- 
#'   custom_metric("top_3_categorical_accuracy", function(y_true, y_pred) {
#'     metric_top_k_categorical_accuracy(y_true, y_pred, k = 3) 
#'   })
#'
#' model %>% compile(
#'   loss = 'categorical_crossentropy',
#'   optimizer = optimizer_rmsprop(),
#'   metrics = metric_top_3_categorical_accuracy
#' )
#' }
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


#' @rdname metric_binary_accuracy
#' @export
custom_metric <- function(name, metric_fn) {
  metric_fn <- reticulate::py_func(metric_fn)
  reticulate::py_set_attr(metric_fn, "__name__", name)
  metric_fn
}











