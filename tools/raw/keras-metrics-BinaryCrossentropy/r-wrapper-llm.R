#' Computes the crossentropy metric between the labels and predictions.
#'
#' @description
#' This is the crossentropy metric class to be used when there are only two
#' label classes (0 and 1).
#'
#' # Examples
#' ```python
#' y_true = [[0, 1], [0, 0]]
#' y_pred = [[0.6, 0.4], [0.4, 0.6]]
#' loss = keras.losses.binary_crossentropy(y_true, y_pred)
#' assert loss.shape == (2,)
#' loss
#' # array([0.916 , 0.714], dtype=float32)
#' ```
#' Standalone usage:
#'
#' ```python
#' m = keras.metrics.BinaryCrossentropy()
#' m.update_state([[0, 1], [0, 0]], [[0.6, 0.4], [0.4, 0.6]])
#' m.result()
#' # 0.81492424
#' ```
#'
#' ```python
#' m.reset_state()
#' m.update_state([[0, 1], [0, 0]], [[0.6, 0.4], [0.4, 0.6]],
#'                sample_weight=[1, 0])
#' m.result()
#' # 0.9162905
#' ```
#'
#' Usage with `compile()` API:
#'
#' ```python
#' model.compile(
#'     optimizer='sgd',
#'     loss='mse',
#'     metrics=[keras.metrics.BinaryCrossentropy()])
#' ```
#'
#' # Returns
#' Binary crossentropy loss value. shape = `[batch_size, d0, .. dN-1]`.
#'
#' @param name (Optional) string name of the metric instance.
#' @param dtype (Optional) data type of the metric result.
#' @param from_logits Whether `y_pred` is expected to be a logits tensor. By
#'     default, we assume that `y_pred` encodes a probability distribution.
#' @param label_smoothing Float in `[0, 1]`. If > `0` then smooth the labels by
#'     squeezing them towards 0.5, that is,
#'     using `1. - 0.5 * label_smoothing` for the target class
#'     and `0.5 * label_smoothing` for the non-target class.
#' @param y_true Ground truth values. shape = `[batch_size, d0, .. dN]`.
#' @param y_pred The predicted values. shape = `[batch_size, d0, .. dN]`.
#' @param axis The axis along which the mean is computed. Defaults to `-1`.
#' @param ... Passed on to the Python callable
#'
#' @export
#' @family metric
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/metrics/BinaryCrossentropy>
metric_binary_crossentropy <-
structure(function (y_true, y_pred, from_logits = FALSE, label_smoothing = 0,
    axis = -1L, ..., name = "binary_crossentropy", dtype = NULL)
{
    args <- capture_args2(list(label_smoothing = as_integer,
        axis = as_axis))
    callable <- if (missing(y_true) && missing(y_pred))
        keras$metrics$BinaryCrossentropy
    else keras$metrics$binary_crossentropy
    do.call(callable, args)
}, py_function_name = "binary_crossentropy")
