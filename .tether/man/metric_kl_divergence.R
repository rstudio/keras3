#' Computes Kullback-Leibler divergence metric between `y_true` and
#'
#' @description
#' Formula:
#'
#' ```python
#' loss = y_true * log(y_true / y_pred)
#' ```
#'
#' `y_pred`.
#'
#' Formula:
#'
#' ```python
#' metric = y_true * log(y_true / y_pred)
#' ```
#'
#' # Usage
#' Standalone usage:
#'
#' ```python
#' m = keras.metrics.KLDivergence()
#' m.update_state([[0, 1], [0, 0]], [[0.6, 0.4], [0.4, 0.6]])
#' m.result()
#' # 0.45814306
#' ```
#'
#' ```python
#' m.reset_state()
#' m.update_state([[0, 1], [0, 0]], [[0.6, 0.4], [0.4, 0.6]],
#'                sample_weight=[1, 0])
#' m.result()
#' # 0.9162892
#' ```
#'
#' Usage with `compile()` API:
#'
#' ```python
#' model.compile(optimizer='sgd',
#'               loss='mse',
#'               metrics=[keras.metrics.KLDivergence()])
#' ```
#'
#' @param name
#' (Optional) string name of the metric instance.
#'
#' @param dtype
#' (Optional) data type of the metric result.
#'
#' @param y_true
#' Tensor of true targets.
#'
#' @param y_pred
#' Tensor of predicted targets.
#'
#' @param ...
#' Passed on to the Python callable
#'
#' @export
#' @family losses
#' @family metrics
#' @family probabilistic metrics
#' @seealso
#' + <https:/keras.io/api/metrics/probabilistic_metrics#kldivergence-class>
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/metrics/KLDivergence>
metric_kl_divergence <-
function (y_true, y_pred, ..., name = "kl_divergence", dtype = NULL)
{
}
