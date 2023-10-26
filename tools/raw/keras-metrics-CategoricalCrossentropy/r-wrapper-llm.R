#' Computes the crossentropy metric between the labels and predictions.
#'
#' @description
#' This is the crossentropy metric class to be used when there are multiple
#' label classes (2 or more). It assumes that labels are one-hot encoded,
#' e.g., when labels values are `[2, 0, 1]`, then
#' `y_true` is `[[0, 0, 1], [1, 0, 0], [0, 1, 0]]`.
#'
#' # Examples
#' ```python
#' y_true = [[0, 1, 0], [0, 0, 1]]
#' y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]
#' loss = keras.losses.categorical_crossentropy(y_true, y_pred)
#' assert loss.shape == (2,)
#' loss
#' # array([0.0513, 2.303], dtype=float32)
#' ```
#' Standalone usage:
#'
#' ```python
#' # EPSILON = 1e-7, y = y_true, y` = y_pred
#' # y` = clip_ops.clip_by_value(output, EPSILON, 1. - EPSILON)
#' # y` = [[0.05, 0.95, EPSILON], [0.1, 0.8, 0.1]]
#' # xent = -sum(y * log(y'), axis = -1)
#' #      = -((log 0.95), (log 0.1))
#' #      = [0.051, 2.302]
#' # Reduced xent = (0.051 + 2.302) / 2
#' m = keras.metrics.CategoricalCrossentropy()
#' m.update_state([[0, 1, 0], [0, 0, 1]],
#'                [[0.05, 0.95, 0], [0.1, 0.8, 0.1]])
#' m.result()
#' # 1.1769392
#' ```
#'
#' ```python
#' m.reset_state()
#' m.update_state([[0, 1, 0], [0, 0, 1]],
#'                [[0.05, 0.95, 0], [0.1, 0.8, 0.1]],
#'                sample_weight=np.array([0.3, 0.7]))
#' m.result()
#' # 1.6271976
#' ```
#'
#' Usage with `compile()` API:
#'
#' ```python
#' model.compile(
#'     optimizer='sgd',
#'     loss='mse',
#'     metrics=[keras.metrics.CategoricalCrossentropy()])
#' ```
#'
#' # Returns
#' Categorical crossentropy loss value.
#'
#' @param name (Optional) string name of the metric instance.
#' @param dtype (Optional) data type of the metric result.
#' @param from_logits Whether `y_pred` is expected to be a logits tensor. By
#'     default, we assume that `y_pred` encodes a probability distribution.
#' @param label_smoothing Float in `[0, 1].` If > `0` then smooth the labels. For
#'     example, if `0.1`, use `0.1 / num_classes` for non-target labels
#'     and `0.9 + 0.1 / num_classes` for target labels.
#' @param axis Defaults to `-1`. The dimension along which the entropy is
#'     computed.
#' @param y_true Tensor of one-hot true targets.
#' @param y_pred Tensor of predicted targets.
#' @param ... Passed on to the Python callable
#'
#' @export
#' @family metric
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/metrics/CategoricalCrossentropy>
metric_categorical_crossentropy <-
structure(function (y_true, y_pred, from_logits = FALSE, label_smoothing = 0,
    axis = -1L, ..., name = "categorical_crossentropy", dtype = NULL)
{
    args <- capture_args2(list(label_smoothing = as_integer,
        axis = as_axis))
    callable <- if (missing(y_true) && missing(y_pred))
        keras$metrics$CategoricalCrossentropy
    else keras$metrics$categorical_crossentropy
    do.call(callable, args)
}, py_function_name = "categorical_crossentropy")
