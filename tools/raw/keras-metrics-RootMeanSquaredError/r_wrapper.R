#' Computes root mean squared error metric between `y_true` and `y_pred`.
#'
#' @description
#' Formula:
#'
#' ```python
#' loss = sqrt(mean((y_pred - y_true) ** 2))
#' ```
#'
#' # Examples
#' Standalone usage:
#'
#' ```python
#' m = keras.metrics.RootMeanSquaredError()
#' m.update_state([[0, 1], [0, 0]], [[1, 1], [0, 0]])
#' m.result()
#' # 0.5
#' ```
#'
#' ```python
#' m.reset_state()
#' m.update_state([[0, 1], [0, 0]], [[1, 1], [0, 0]],
#'                sample_weight=[1, 0])
#' m.result()
#' # 0.70710677
#' ```
#'
#' Usage with `compile()` API:
#'
#' ```python
#' model.compile(
#'     optimizer='sgd',
#'     loss='mse',
#'     metrics=[keras.metrics.RootMeanSquaredError()])
#' ```
#'
#' @param name (Optional) string name of the metric instance.
#' @param dtype (Optional) data type of the metric result.
#' @param ... Passed on to the Python callable
#'
#' @export
#' @family metric
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/metrics/RootMeanSquaredError>
metric_root_mean_squared_error <-
function (..., name = "root_mean_squared_error", dtype = NULL)
{
    args <- capture_args2(NULL)
    do.call(keras$metrics$RootMeanSquaredError, args)
}
