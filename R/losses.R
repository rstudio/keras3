
#' Model loss functions
#' 
#' @param y_true True labels (Tensor) 
#' @param y_pred Predictions (Tensor of the same shape as `y_true`)
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
#' - `y_true` True labels (Tensor) 
#' - `y_pred` Predictions (Tensor of the same shape as `y_true`)
#' 
#' The actual optimized objective is the mean of the output array across all
#' datapoints.
#' 
#' @section Categorical Crossentropy:
#'   
#'   When using the categorical_crossentropy loss, your targets should be in
#'   categorical format (e.g. if you have 10 classes, the target for each sample
#'   should be a 10-dimensional vector that is all-zeros except for a 1 at the
#'   index corresponding to the class of the sample). In order to convert
#'   integer targets into categorical targets, you can use the Keras utility
#'   function [to_categorical()]:
#'   
#'   `categorical_labels <- to_categorical(int_labels, num_classes = NULL)`
#'   
#' @section loss_logcosh:
#' 
#' `log(cosh(x))` is approximately equal to `(x ** 2) / 2` for small `x` and
#' to `abs(x) - log(2)` for large `x`. This means that 'logcosh' works mostly
#' like the mean squared error, but will not be so strongly affected by the
#' occasional wildly incorrect prediction. However, it may return NaNs if the
#' intermediate value `cosh(y_pred - y_true)` is too large to be represented
#' in the chosen precision.   
#'   
#' @seealso [compile()]   
#'   
#' @export
loss_mean_squared_error <- function(y_true, y_pred) {
  keras$losses$mean_squared_error(y_true, y_pred)
}
attr(loss_mean_squared_error, "py_function_name") <- "mean_squared_error"

#' @rdname loss_mean_squared_error
#' @export
loss_mean_absolute_error <- function(y_true, y_pred) {
  keras$losses$mean_absolute_error(y_true, y_pred)
}
attr(loss_mean_absolute_error, "py_function_name") <- "mean_absolute_error"

#' @rdname loss_mean_squared_error
#' @export
loss_mean_absolute_percentage_error <- function(y_true, y_pred) {
  keras$losses$mean_absolute_percentage_error(y_true, y_pred)
}
attr(loss_mean_absolute_percentage_error, "py_function_name") <- "mean_absolute_percentage_error"

#' @rdname loss_mean_squared_error
#' @export
loss_mean_squared_logarithmic_error <- function(y_true, y_pred) {
  keras$losses$mean_squared_logarithmic_error(y_true, y_pred)
}
attr(loss_mean_squared_logarithmic_error, "py_function_name") <- "mean_squared_logarithmic_error"

#' @rdname loss_mean_squared_error
#' @export
loss_squared_hinge <- function(y_true, y_pred) {
  keras$losses$squared_hinge(y_true, y_pred)
}
attr(loss_squared_hinge, "py_function_name") <- "squared_hinge"

#' @rdname loss_mean_squared_error
#' @export
loss_hinge <- function(y_true, y_pred) {
  keras$losses$hinge(y_true, y_pred)
}
attr(loss_hinge, "py_function_name") <- "hinge"

#' @rdname loss_mean_squared_error
#' @export
loss_categorical_hinge <- function(y_true, y_pred) {
  keras$losses$categorical_hinge(y_true, y_pred)
}
attr(loss_hinge, "py_function_name") <- "categorical_hinge"

#' @rdname loss_mean_squared_error
#' @export
loss_logcosh <- function(y_true, y_pred) {
  keras$losses$logcosh(y_true, y_pred)
}
attr(loss_hinge, "py_function_name") <- "logcosh"


#' @rdname loss_mean_squared_error
#' @export
loss_categorical_crossentropy <- function(y_true, y_pred) {
  keras$losses$categorical_crossentropy(y_true, y_pred)
}
attr(loss_categorical_crossentropy, "py_function_name") <- "categorical_crossentropy"


#' @rdname loss_mean_squared_error
#' @export
loss_sparse_categorical_crossentropy <- function(y_true, y_pred) {
  keras$losses$sparse_categorical_crossentropy(y_true, y_pred)
}
attr(loss_sparse_categorical_crossentropy, "py_function_name") <- "sparse_categorical_crossentropy"

#' @rdname loss_mean_squared_error
#' @export
loss_binary_crossentropy <- function(y_true, y_pred) {
  keras$losses$binary_crossentropy(y_true, y_pred)
}
attr(loss_binary_crossentropy, "py_function_name") <- "binary_crossentropy"

#' @rdname loss_mean_squared_error
#' @export
loss_kullback_leibler_divergence <- function(y_true, y_pred) {
  keras$losses$kullback_leibler_divergence(y_true, y_pred)
}
attr(loss_kullback_leibler_divergence, "py_function_name") <- "kullback_leibler_divergence"

#' @rdname loss_mean_squared_error
#' @export
loss_poisson <- function(y_true, y_pred) {
  keras$losses$poisson(y_true, y_pred)
}
attr(loss_poisson, "py_function_name") <- "poisson"

#' @rdname loss_mean_squared_error
#' @export
loss_cosine_proximity <- function(y_true, y_pred) {
  keras$losses$cosine_proximity(y_true, y_pred)
}
attr(loss_cosine_proximity, "py_function_name") <- "cosine_proximity"





