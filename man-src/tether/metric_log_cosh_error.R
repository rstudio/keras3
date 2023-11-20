#' Computes the logarithm of the hyperbolic cosine of the prediction error.
#'
#' @description
#' Formula:
#'
#' ```python
#' error = y_pred - y_true
#' logcosh = mean(log((exp(error) + exp(-error))/2), axis=-1)
#' ```
#'
#' # Examples
#' Standalone usage:
#'
#' ```python
#' m = keras.metrics.LogCoshError()
#' m.update_state([[0, 1], [0, 0]], [[1, 1], [0, 0]])
#' m.result()
#' # 0.10844523
#' m.reset_state()
#' m.update_state([[0, 1], [0, 0]], [[1, 1], [0, 0]],
#'                sample_weight=[1, 0])
#' m.result()
#' # 0.21689045
#' ```
#'
#' Usage with `compile()` API:
#'
#' ```python
#' model.compile(optimizer='sgd',
#'               loss='mse',
#'               metrics=[keras.metrics.LogCoshError()])
#' ```
#'
#' @param name
#' (Optional) string name of the metric instance.
#'
#' @param dtype
#' (Optional) data type of the metric result.
#'
#' @param ...
#' Passed on to the Python callable
#'
#' @export
#' @family regression metrics
#' @family metrics
#' @seealso
#' + <https:/keras.io/api/metrics/regression_metrics#logcosherror-class>
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/metrics/LogCoshError>
metric_log_cosh_error <-
function (..., name = "logcosh", dtype = NULL)
{
}
