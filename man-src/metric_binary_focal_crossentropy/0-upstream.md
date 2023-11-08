keras.metrics.binary_focal_crossentropy
__signature__
(
  y_true,
  y_pred,
  apply_class_balancing=False,
  alpha=0.25,
  gamma=2.0,
  from_logits=False,
  label_smoothing=0.0,
  axis=-1
)
__doc__
Computes the binary focal crossentropy loss.

According to [Lin et al., 2018](https://arxiv.org/pdf/1708.02002.pdf), it
helps to apply a focal factor to down-weight easy examples and focus more on
hard examples. By default, the focal tensor is computed as follows:

`focal_factor = (1 - output) ** gamma` for class 1
`focal_factor = output ** gamma` for class 0
where `gamma` is a focusing parameter. When `gamma` = 0, there is no focal
effect on the binary crossentropy loss.

If `apply_class_balancing == True`, this function also takes into account a
weight balancing factor for the binary classes 0 and 1 as follows:

`weight = alpha` for class 1 (`target == 1`)
`weight = 1 - alpha` for class 0
where `alpha` is a float in the range of `[0, 1]`.

Args:
    y_true: Ground truth values, of shape `(batch_size, d0, .. dN)`.
    y_pred: The predicted values, of shape `(batch_size, d0, .. dN)`.
    apply_class_balancing: A bool, whether to apply weight balancing on the
        binary classes 0 and 1.
    alpha: A weight balancing factor for class 1, default is `0.25` as
        mentioned in the reference. The weight for class 0 is `1.0 - alpha`.
    gamma: A focusing parameter, default is `2.0` as mentioned in the
        reference.
    from_logits: Whether `y_pred` is expected to be a logits tensor. By
        default, we assume that `y_pred` encodes a probability distribution.
    label_smoothing: Float in `[0, 1]`. If > `0` then smooth the labels by
        squeezing them towards 0.5, that is,
        using `1. - 0.5 * label_smoothing` for the target class
        and `0.5 * label_smoothing` for the non-target class.
    axis: The axis along which the mean is computed. Defaults to `-1`.

Returns:
    Binary focal crossentropy loss value
    with shape = `[batch_size, d0, .. dN-1]`.

Example:

>>> y_true = [[0, 1], [0, 0]]
>>> y_pred = [[0.6, 0.4], [0.4, 0.6]]
>>> loss = keras.losses.binary_focal_crossentropy(
...        y_true, y_pred, gamma=2)
>>> assert loss.shape == (2,)
>>> loss
array([0.330, 0.206], dtype=float32)
