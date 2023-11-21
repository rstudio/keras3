#' Computes how often integer targets are in the top `K` predictions.
#'
#' @description
#'
#' # Usage
#' Standalone usage:
#'
#' ```python
#' m = keras.metrics.SparseTopKCategoricalAccuracy(k=1)
#' m.update_state([2, 1], [[0.1, 0.9, 0.8], [0.05, 0.95, 0]])
#' m.result()
#' # 0.5
#' ```
#'
#' ```python
#' m.reset_state()
#' m.update_state([2, 1], [[0.1, 0.9, 0.8], [0.05, 0.95, 0]],
#'                sample_weight=[0.7, 0.3])
#' m.result()
#' # 0.3
#' ```
#'
#' Usage with `compile()` API:
#'
#' ```python
#' model.compile(optimizer='sgd',
#'               loss='sparse_categorical_crossentropy',
#'               metrics=[keras.metrics.SparseTopKCategoricalAccuracy()])
#' ```
#'
#' @param k
#' (Optional) Number of top elements to look at for computing accuracy.
#' Defaults to `5`.
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
#' @family accuracy metrics
#' @family metrics
#' @seealso
#' + <https:/keras.io/api/metrics/accuracy_metrics#sparsetopkcategoricalaccuracy-class>
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/metrics/SparseTopKCategoricalAccuracy>
metric_sparse_top_k_categorical_accuracy <-
function (y_true, y_pred, k = 5L, ..., name = "sparse_top_k_categorical_accuracy",
    dtype = NULL)
{
}
