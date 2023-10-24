#' Computes the mean squared error between `y_true` and `y_pred`.
#'
#' @description
#' Formula:
#'
#' ```python
#' loss = mean(square(y_true - y_pred), axis=-1)
#' ```
#' Formula:
#'
#' ```python
#' loss = mean(square(y_true - y_pred))
#' ```
#'
#' # Examples
#' ```python
#' y_true = np.random.randint(0, 2, size=(2, 3))
#' y_pred = np.random.random(size=(2, 3))
#' loss = keras.losses.mean_squared_error(y_true, y_pred)
#' ```
#' ```python
#' m = keras.metrics.MeanSquaredError()
#' m.update_state([[0, 1], [0, 0]], [[1, 1], [0, 0]])
#' m.result()
#' # 0.25
#' ```
#'
#' # Returns
#'     Mean squared error values with shape = `[batch_size, d0, .. dN-1]`.
#'
#' @param name (Optional) string name of the metric instance.
#' @param dtype (Optional) data type of the metric result.
#' @param y_true Ground truth values with shape = `[batch_size, d0, .. dN]`.
#' @param y_pred The predicted values with shape = `[batch_size, d0, .. dN]`.
#' @param ... Passed on to the Python callable
#'
#' @export
#' @family metric
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/metrics/MeanSquaredError>
metric_mean_squared_error <-
structure(function (y_true, y_pred, ..., name = "mean_squared_error",
    dtype = NULL)
{
    args <- capture_args2(NULL)
    callable <- if (missing(y_true) && missing(y_pred))
        keras$metrics$MeanSquaredError
    else keras$metrics$mean_squared_error
    do.call(callable, args)
}, py_function_name = "mean_squared_error")
