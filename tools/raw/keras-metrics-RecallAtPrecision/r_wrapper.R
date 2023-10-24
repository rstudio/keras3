#' Computes best recall where precision is >= specified value.
#'
#' @description
#' For a given score-label-distribution the required precision might not
#' be achievable, in this case 0.0 is returned as recall.
#'
#' This metric creates four local variables, `true_positives`,
#' `true_negatives`, `false_positives` and `false_negatives` that are used to
#' compute the recall at the given precision. The threshold for the given
#' precision value is computed and used to evaluate the corresponding recall.
#'
#' If `sample_weight` is `None`, weights default to 1.
#' Use `sample_weight` of 0 to mask values.
#'
#' If `class_id` is specified, we calculate precision by considering only the
#' entries in the batch for which `class_id` is above the threshold
#' predictions, and computing the fraction of them for which `class_id` is
#' indeed a correct label.
#'
#' # Usage
#' Standalone usage:
#'
#' ```python
#' m = keras.metrics.RecallAtPrecision(0.8)
#' m.update_state([0, 0, 1, 1], [0, 0.5, 0.3, 0.9])
#' m.result()
#' # 0.5
#' ```
#'
#' ```python
#' m.reset_state()
#' m.update_state([0, 0, 1, 1], [0, 0.5, 0.3, 0.9],
#'                sample_weight=[1, 0, 0, 1])
#' m.result()
#' # 1.0
#' ```
#'
#' Usage with `compile()` API:
#'
#' ```python
#' model.compile(
#'     optimizer='sgd',
#'     loss='mse',
#'     metrics=[keras.metrics.RecallAtPrecision(precision=0.8)])
#' ```
#'
#' @param precision A scalar value in range `[0, 1]`.
#' @param num_thresholds (Optional) Defaults to 200. The number of thresholds
#'     to use for matching the given precision.
#' @param class_id (Optional) Integer class ID for which we want binary metrics.
#'     This must be in the half-open interval `[0, num_classes)`, where
#'     `num_classes` is the last dimension of predictions.
#' @param name (Optional) string name of the metric instance.
#' @param dtype (Optional) data type of the metric result.
#' @param ... Passed on to the Python callable
#'
#' @export
#' @family metric
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/metrics/RecallAtPrecision>
metric_recall_at_precision <-
function (..., precision, num_thresholds = 200L, class_id = NULL,
    name = NULL, dtype = NULL)
{
    args <- capture_args2(list(num_thresholds = as_integer, class_id = as_integer))
    do.call(keras$metrics$RecallAtPrecision, args)
}
