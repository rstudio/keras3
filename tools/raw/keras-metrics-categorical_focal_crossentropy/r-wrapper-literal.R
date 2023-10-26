#' Computes the categorical focal crossentropy loss.
#'
#' @description
#'
#' # Returns
#' Categorical focal crossentropy loss value.
#'
#' # Examples
#' ```python
#' y_true = [[0, 1, 0], [0, 0, 1]]
#' y_pred = [[0.05, 0.9, 0.05], [0.1, 0.85, 0.05]]
#' loss = keras.losses.categorical_focal_crossentropy(y_true, y_pred)
#' assert loss.shape == (2,)
#' loss
#' # array([2.63401289e-04, 6.75912094e-01], dtype=float32)
#' ```
#'
#' @param y_true Tensor of one-hot true targets.
#' @param y_pred Tensor of predicted targets.
#' @param alpha A weight balancing factor for all classes, default is `0.25` as
#'     mentioned in the reference. It can be a list of floats or a scalar.
#'     In the multi-class case, alpha may be set by inverse class
#'     frequency by using `compute_class_weight` from `sklearn.utils`.
#' @param gamma A focusing parameter, default is `2.0` as mentioned in the
#'     reference. It helps to gradually reduce the importance given to
#'     simple examples in a smooth manner. When `gamma` = 0, there is
#'     no focal effect on the categorical crossentropy.
#' @param from_logits Whether `y_pred` is expected to be a logits tensor. By
#'     default, we assume that `y_pred` encodes a probability
#'     distribution.
#' @param label_smoothing Float in `[0, 1].` If > `0` then smooth the labels. For
#'     example, if `0.1`, use `0.1 / num_classes` for non-target labels
#'     and `0.9 + 0.1 / num_classes` for target labels.
#' @param axis Defaults to `-1`. The dimension along which the entropy is
#'     computed.
#'
#' @export
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/metrics/categorical_focal_crossentropy>
metric_categorical_focal_crossentropy <-
function (y_true, y_pred, alpha = 0.25, gamma = 2, from_logits = FALSE,
    label_smoothing = 0, axis = -1L)
{
    args <- capture_args2(list(axis = as_axis))
    do.call(keras$metrics$categorical_focal_crossentropy, args)
}
