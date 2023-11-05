Computes the categorical focal crossentropy loss.

Args:
    y_true: Tensor of one-hot true targets.
    y_pred: Tensor of predicted targets.
    alpha: A weight balancing factor for all classes, default is `0.25` as
        mentioned in the reference. It can be a list of floats or a scalar.
        In the multi-class case, alpha may be set by inverse class
        frequency by using `compute_class_weight` from `sklearn.utils`.
    gamma: A focusing parameter, default is `2.0` as mentioned in the
        reference. It helps to gradually reduce the importance given to
        simple examples in a smooth manner. When `gamma` = 0, there is
        no focal effect on the categorical crossentropy.
    from_logits: Whether `y_pred` is expected to be a logits tensor. By
        default, we assume that `y_pred` encodes a probability
        distribution.
    label_smoothing: Float in [0, 1]. If > `0` then smooth the labels. For
        example, if `0.1`, use `0.1 / num_classes` for non-target labels
        and `0.9 + 0.1 / num_classes` for target labels.
    axis: Defaults to `-1`. The dimension along which the entropy is
        computed.

Returns:
    Categorical focal crossentropy loss value.

Example:

>>> y_true = [[0, 1, 0], [0, 0, 1]]
>>> y_pred = [[0.05, 0.9, 0.05], [0.1, 0.85, 0.05]]
>>> loss = keras.losses.categorical_focal_crossentropy(y_true, y_pred)
>>> assert loss.shape == (2,)
>>> loss
array([2.63401289e-04, 6.75912094e-01], dtype=float32)
