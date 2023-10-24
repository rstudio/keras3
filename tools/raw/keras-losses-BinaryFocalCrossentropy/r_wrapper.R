#' Computes focal cross-entropy loss between true labels and predictions.
#'
#' @description
#' According to [Lin et al., 2018](https://arxiv.org/pdf/1708.02002.pdf), it
#' helps to apply a focal factor to down-weight easy examples and focus more on
#' hard examples. By default, the focal tensor is computed as follows:
#'
#' `focal_factor = (1 - output) ** gamma` for class 1
#' `focal_factor = output ** gamma` for class 0
#' where `gamma` is a focusing parameter. When `gamma` = 0, there is no focal
#' effect on the binary crossentropy loss.
#'
#' If `apply_class_balancing == True`, this function also takes into account a
#' weight balancing factor for the binary classes 0 and 1 as follows:
#'
#' `weight = alpha` for class 1 (`target == 1`)
#' `weight = 1 - alpha` for class 0
#' where `alpha` is a float in the range of `[0, 1]`.
#' Binary cross-entropy loss is often used for binary (0 or 1) classification
#' tasks. The loss function requires the following inputs:
#'
#' - `y_true` (true label): This is either 0 or 1.
#' - `y_pred` (predicted value): This is the model's prediction, i.e, a single
#'     floating-point value which either represents a
#'     `[logit](https://en.wikipedia.org/wiki/Logit), (i.e, value in [-inf, inf]`
#'     when `from_logits=True`) or a probability (i.e, value in `[0., 1.]` when
#'     `from_logits=False`).
#'
#' According to [Lin et al., 2018](https://arxiv.org/pdf/1708.02002.pdf), it
#' helps to apply a "focal factor" to down-weight easy examples and focus more
#' on hard examples. By default, the focal tensor is computed as follows:
#'
#' `focal_factor = (1 - output) ** gamma` for class 1
#' `focal_factor = output ** gamma` for class 0
#' where `gamma` is a focusing parameter. When `gamma=0`, this function is
#' equivalent to the binary crossentropy loss.
#'
#' # Examples
#' ```python
#' y_true = [[0, 1], [0, 0]]
#' y_pred = [[0.6, 0.4], [0.4, 0.6]]
#' loss = keras.losses.binary_focal_crossentropy(
#'        y_true, y_pred, gamma=2)
#' assert loss.shape == (2,)
#' loss
#' # array([0.330, 0.206], dtype=float32)
#' ```
#' With the `compile()` API:
#'
#' ```python
#' model.compile(
#'     loss=keras.losses.BinaryFocalCrossentropy(
#'         gamma=2.0, from_logits=True),
#'     ...
#' )
#' ```
#'
#' As a standalone function:
#'
#' ```python
#' # Example 1: (batch_size = 1, number of samples = 4)
#' y_true = [0, 1, 0, 0]
#' y_pred = [-18.6, 0.51, 2.94, -12.8]
#' loss = keras.losses.BinaryFocalCrossentropy(
#'    gamma=2, from_logits=True)
#' loss(y_true, y_pred)
#' # 0.691
#' ```
#'
#' ```python
#' # Apply class weight
#' loss = keras.losses.BinaryFocalCrossentropy(
#'     apply_class_balancing=True, gamma=2, from_logits=True)
#' loss(y_true, y_pred)
#' # 0.51
#' ```
#'
#' ```python
#' # Example 2: (batch_size = 2, number of samples = 4)
#' y_true = [[0, 1], [0, 0]]
#' y_pred = [[-18.6, 0.51], [2.94, -12.8]]
#' # Using default 'auto'/'sum_over_batch_size' reduction type.
#' loss = keras.losses.BinaryFocalCrossentropy(
#'     gamma=3, from_logits=True)
#' loss(y_true, y_pred)
#' # 0.647
#' ```
#'
#' ```python
#' # Apply class weight
#' loss = keras.losses.BinaryFocalCrossentropy(
#'      apply_class_balancing=True, gamma=3, from_logits=True)
#' loss(y_true, y_pred)
#' # 0.482
#' ```
#'
#' ```python
#' # Using 'sample_weight' attribute with focal effect
#' loss = keras.losses.BinaryFocalCrossentropy(
#'     gamma=3, from_logits=True)
#' loss(y_true, y_pred, sample_weight=[0.8, 0.2])
#' # 0.133
#' ```
#'
#' ```python
#' # Apply class weight
#' loss = keras.losses.BinaryFocalCrossentropy(
#'      apply_class_balancing=True, gamma=3, from_logits=True)
#' loss(y_true, y_pred, sample_weight=[0.8, 0.2])
#' # 0.097
#' ```
#'
#' ```python
#' # Using 'sum' reduction` type.
#' loss = keras.losses.BinaryFocalCrossentropy(
#'     gamma=4, from_logits=True,
#'     reduction="sum")
#' loss(y_true, y_pred)
#' # 1.222
#' ```
#'
#' ```python
#' # Apply class weight
#' loss = keras.losses.BinaryFocalCrossentropy(
#'     apply_class_balancing=True, gamma=4, from_logits=True,
#'     reduction="sum")
#' loss(y_true, y_pred)
#' # 0.914
#' ```
#'
#' ```python
#' # Using 'none' reduction type.
#' loss = keras.losses.BinaryFocalCrossentropy(
#'     gamma=5, from_logits=True,
#'     reduction=None)
#' loss(y_true, y_pred)
#' # array([0.0017 1.1561], dtype=float32)
#' ```
#'
#' ```python
#' # Apply class weight
#' loss = keras.losses.BinaryFocalCrossentropy(
#'     apply_class_balancing=True, gamma=5, from_logits=True,
#'     reduction=None)
#' loss(y_true, y_pred)
#' # array([0.0004 0.8670], dtype=float32)
#' ```
#'
#' # Returns
#' Binary focal crossentropy loss value
#' with shape = `[batch_size, d0, .. dN-1]`.
#'
#' @param apply_class_balancing A bool, whether to apply weight balancing on the
#'     binary classes 0 and 1.
#' @param alpha A weight balancing factor for class 1, default is `0.25` as
#'     mentioned in the reference. The weight for class 0 is `1.0 - alpha`.
#' @param gamma A focusing parameter, default is `2.0` as mentioned in the
#'     reference.
#' @param from_logits Whether `y_pred` is expected to be a logits tensor. By
#'     default, we assume that `y_pred` encodes a probability distribution.
#' @param label_smoothing Float in `[0, 1]`. If > `0` then smooth the labels by
#'     squeezing them towards 0.5, that is,
#'     using `1. - 0.5 * label_smoothing` for the target class
#'     and `0.5 * label_smoothing` for the non-target class.
#' @param axis The axis along which the mean is computed. Defaults to `-1`.
#' @param reduction Type of reduction to apply to the loss. In almost all cases
#'     this should be `"sum_over_batch_size"`.
#'     Supported options are `"sum"`, `"sum_over_batch_size"` or `None`.
#' @param name Optional name for the loss instance.
#' @param y_true Ground truth values, of shape `(batch_size, d0, .. dN)`.
#' @param y_pred The predicted values, of shape `(batch_size, d0, .. dN)`.
#' @param ... Passed on to the Python callable
#'
#' @export
#' @family loss
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/losses/BinaryFocalCrossentropy>
loss_binary_focal_crossentropy <-
structure(function (y_true, y_pred, apply_class_balancing = FALSE,
    alpha = 0.25, gamma = 2, from_logits = FALSE, label_smoothing = 0,
    axis = -1L, ..., reduction = "sum_over_batch_size", name = "binary_focal_crossentropy")
{
    args <- capture_args2(list(axis = as_axis))
    callable <- if (missing(y_true) && missing(y_pred))
        keras$losses$BinaryFocalCrossentropy
    else keras$losses$binary_focal_crossentropy
    do.call(callable, args)
}, py_function_name = "binary_focal_crossentropy")
