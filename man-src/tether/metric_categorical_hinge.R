#' Computes the categorical hinge metric between `y_true` and `y_pred`.
#'
#' @description
#' Formula:
#'
#' ```python
#' loss = maximum(neg - pos + 1, 0)
#' ```
#'
#' where `neg=maximum((1-y_true)*y_pred)` and `pos=sum(y_true*y_pred)`
#'
#' # Usage
#' Standalone usage:
#' ```python
#' m = keras.metrics.CategoricalHinge()
#' m.update_state([[0, 1], [0, 0]], [[0.6, 0.4], [0.4, 0.6]])
#' m.result().numpy()
#' # 1.4000001
#' m.reset_state()
#' m.update_state([[0, 1], [0, 0]], [[0.6, 0.4], [0.4, 0.6]],
#'                sample_weight=[1, 0])
#' m.result()
#' # 1.2
#' ```
#'
#' @param name
#' (Optional) string name of the metric instance.
#'
#' @param dtype
#' (Optional) data type of the metric result.
#'
#' @param y_true
#' The ground truth values. `y_true` values are expected to be
#' either `{-1, +1}` or `{0, 1}` (i.e. a one-hot-encoded tensor) with
#' shape = `[batch_size, d0, .. dN]`.
#'
#' @param y_pred
#' The predicted values with shape = `[batch_size, d0, .. dN]`.
#'
#' @param ...
#' Passed on to the Python callable
#'
#' @export
#' @family losses
#' @family metrics
#' @family hinge metrics
#' @seealso
#' + <https:/keras.io/api/metrics/hinge_metrics#categoricalhinge-class>
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/metrics/CategoricalHinge>
metric_categorical_hinge <-
function (y_true, y_pred, ..., name = "categorical_hinge", dtype = NULL)
{
}
