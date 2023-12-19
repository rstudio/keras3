
#' Metric
#'
#' A `Metric` object encapsulates metric logic and state that can be used to
#' track model performance during training. It is what is returned by the family
#' of metric functions that start with prefix `metric_*`.
#'
#' @param name (Optional) string name of the metric instance.
#' @param dtype (Optional) data type of the metric result.
#'
#' @returns A (subclassed) `Metric` instance that can be passed directly to
#'   `compile(metrics = )`, or used as a standalone object. See `?Metric` for
#'   example usage.
#'
#'
#' @section Usage with `compile`:
#' ```r
#' model %>% compile(
#'   optimizer = 'sgd',
#'   loss = 'mse',
#'   metrics = list(metric_SOME_METRIC(), metric_SOME_OTHER_METRIC())
#' )
#' ```
#'
#' @section Standalone usage:
#' ```r
#' m <- metric_SOME_METRIC()
#' for (e in seq(epochs)) {
#'   for (i in seq(train_steps)) {
#'     c(y_true, y_pred, sample_weight = NULL) %<-% ...
#'     m$update_state(y_true, y_pred, sample_weight)
#'   }
#'   cat('Final epoch result: ', as.numeric(m$result()), "\n")
#'   m$reset_state()
#' }
#' ```
#'
#' @section Custom Metric (subclass):
#' To be implemented by subclasses:
#'
#'   *  `initialize()`: All state variables should be created in this method by calling `self$add_weight()` like:
#'
#'          self$var <- self$add_weight(...)
#'
#'   *  `update_state()`: Has all updates to the state variables like:
#'
#'          self$var$assign_add(...)
#'
#'   *  `result()`: Computes and returns a value for the metric from the state variables.
#'
#' Example custom metric subclass:
#' ````R
#' metric_binary_true_positives <- new_metric_class(
#'   classname = "BinaryTruePositives",
#'   initialize = function(name = 'binary_true_positives', ...) {
#'     super$initialize(name = name, ...)
#'     self$true_positives <-
#'       self$add_weight(name = 'tp', initializer = 'zeros')
#'   },
#'
#'   update_state = function(y_true, y_pred, sample_weight = NULL) {
#'     y_true <- k_cast(y_true, "bool")
#'     y_pred <- k_cast(y_pred, "bool")
#'
#'     values <- y_true & y_pred
#'     values <- k_cast(values, self$dtype)
#'     if (!is.null(sample_weight)) {
#'       sample_weight <- k_cast(sample_weight, self$dtype)
#'       sample_weight <- tf$broadcast_to(sample_weight, values$shape)
#'       values <- values * sample_weight
#'     }
#'     self$true_positives$assign_add(tf$reduce_sum(values))
#'   },
#'
#'   result = function()
#'     self$true_positives
#' )
#' model %>% compile(..., metrics = list(metric_binary_true_positives()))
#' ````
#' The same `metric_binary_true_positives` could be built with `%py_class%` like
#' this:
#' ````r
#' metric_binary_true_positives(keras$metrics$Metric) %py_class% {
#'   initialize <- <same-as-above>,
#'   update_state <- <same-as-above>,
#'   result <- <same-as-above>
#' }
#' ````
#'
#' @name Metric
#' @rdname Metric
NULL


#' @title metric-or-Metric
#' @name metric-or-Metric
#' @rdname metric-or-Metric
#' @keywords internal
#'
#' @param y_true Tensor of true targets.
#' @param y_pred Tensor of predicted targets.
#' @param ... Passed on to the underlying metric. Used for forwards and backwards compatibility.
#' @param axis (Optional) (1-based) Defaults to -1. The dimension along which the metric is computed.
#' @param name (Optional) string name of the metric instance.
#' @param dtype (Optional) data type of the metric result.
#'
#' @returns If `y_true` and `y_pred` are missing, a (subclassed) `Metric`
#'   instance is returned. The `Metric` object can be passed directly to
#'   `compile(metrics = )` or used as a standalone object. See `?Metric` for
#'   example usage.
#'
#'   Alternatively, if called with `y_true` and `y_pred` arguments, then the
#'   computed case-wise values for the mini-batch are returned directly.
NULL



#' Custom metric function
#'
#' @param name name used to show training progress output
#' @param metric_fn An R function with signature `function(y_true, y_pred){}` that accepts tensors.
#'
#' @details
#' You can provide an arbitrary R function as a custom metric. Note that
#' the `y_true` and `y_pred` parameters are tensors, so computations on
#' them should use backend tensor functions.
#'
#' Use the `custom_metric()` function to define a custom metric.
#' Note that a name ('mean_pred') is provided for the custom metric
#' function: this name is used within training progress output.
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
#' found at <https://tensorflow.rstudio.com/reference/keras/#backend>.
#'
#' Alternative ways of supplying custom metrics:
#'  +  `custom_metric():` Arbitrary R function.
#'  +  [metric_mean_wrapper()]: Wrap an arbitrary R function in a `Metric` instance.
#'  +  subclass `keras$metrics$Metric`: see `?Metric` for example.
#'
#' @family metrics
#' @export
custom_metric <- function(name, metric_fn) {
  metric_fn <- reticulate::py_func(metric_fn)
  reticulate::py_set_attr(metric_fn, "__name__", name)
  metric_fn
}

# can be used w/ activations, regularizers, metrics, loss, anything else
# where it helps to have a name
custom_fn <- function(name, fn) {
  py_func2(fn, TRUE, name)
}



#   cat("@inheritParams Metric")
#   cat("@inherit Metric return")
