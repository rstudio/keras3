keras.metrics.Recall
__signature__
(
  thresholds=None,
  top_k=None,
  class_id=None,
  name=None,
  dtype=None
)
__doc__
Computes the recall of the predictions with respect to the labels.

This metric creates two local variables, `true_positives` and
`false_negatives`, that are used to compute the recall. This value is
ultimately returned as `recall`, an idempotent operation that simply divides
`true_positives` by the sum of `true_positives` and `false_negatives`.

If `sample_weight` is `None`, weights default to 1.
Use `sample_weight` of 0 to mask values.

If `top_k` is set, recall will be computed as how often on average a class
among the labels of a batch entry is in the top-k predictions.

If `class_id` is specified, we calculate recall by considering only the
entries in the batch for which `class_id` is in the label, and computing the
fraction of them for which `class_id` is above the threshold and/or in the
top-k predictions.

Args:
    thresholds: (Optional) A float value, or a Python list/tuple of float
        threshold values in `[0, 1]`. A threshold is compared with
        prediction values to determine the truth value of predictions (i.e.,
        above the threshold is `True`, below is `False`). If used with a
        loss function that sets `from_logits=True` (i.e. no sigmoid
        applied to predictions), `thresholds` should be set to 0.
        One metric value is generated for each threshold value.
        If neither `thresholds` nor `top_k` are set,
        the default is to calculate recall with `thresholds=0.5`.
    top_k: (Optional) Unset by default. An int value specifying the top-k
        predictions to consider when calculating recall.
    class_id: (Optional) Integer class ID for which we want binary metrics.
        This must be in the half-open interval `[0, num_classes)`, where
        `num_classes` is the last dimension of predictions.
    name: (Optional) string name of the metric instance.
    dtype: (Optional) data type of the metric result.

Usage:

Standalone usage:

>>> m = keras.metrics.Recall()
>>> m.update_state([0, 1, 1, 1], [1, 0, 1, 1])
>>> m.result()
0.6666667

>>> m.reset_state()
>>> m.update_state([0, 1, 1, 1], [1, 0, 1, 1], sample_weight=[0, 0, 1, 0])
>>> m.result()
1.0

Usage with `compile()` API:

```python
model.compile(optimizer='sgd',
              loss='mse',
              metrics=[keras.metrics.Recall()])
```

Usage with a loss with `from_logits=True`:

```python
model.compile(optimizer='adam',
              loss=keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=[keras.metrics.Recall(thresholds=0)])
```
