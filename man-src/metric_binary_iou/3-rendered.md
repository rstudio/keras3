Computes the Intersection-Over-Union metric for class 0 and/or 1.

@description
Formula:

```python
iou = true_positives / (true_positives + false_positives + false_negatives)
```
Intersection-Over-Union is a common evaluation metric for semantic image
segmentation.

To compute IoUs, the predictions are accumulated in a confusion matrix,
weighted by `sample_weight` and the metric is then calculated from it.

If `sample_weight` is `None`, weights default to 1.
Use `sample_weight` of 0 to mask values.

This class can be used to compute IoUs for a binary classification task
where the predictions are provided as logits. First a `threshold` is applied
to the predicted values such that those that are below the `threshold` are
converted to class 0 and those that are above the `threshold` are converted
to class 1.

IoUs for classes 0 and 1 are then computed, the mean of IoUs for the classes
that are specified by `target_class_ids` is returned.

# Note
with `threshold=0`, this metric has the same behavior as `IoU`.

# Examples
Standalone usage:

```python
m = keras.metrics.BinaryIoU(target_class_ids=[0, 1], threshold=0.3)
m.update_state([0, 1, 0, 1], [0.1, 0.2, 0.4, 0.7])
m.result()
# 0.33333334
```

```python
m.reset_state()
m.update_state([0, 1, 0, 1], [0.1, 0.2, 0.4, 0.7],
               sample_weight=[0.2, 0.3, 0.4, 0.1])
# cm = [[0.2, 0.4],
#        [0.3, 0.1]]
# sum_row = [0.6, 0.4], sum_col = [0.5, 0.5],
# true_positives = [0.2, 0.1]
# iou = [0.222, 0.125]
m.result()
# 0.17361112
```

Usage with `compile()` API:

```python
model.compile(
    optimizer='sgd',
    loss='mse',
    metrics=[keras.metrics.BinaryIoU(
        target_class_ids=[0],
        threshold=0.5
    )]
)
```

@param target_class_ids
A tuple or list of target class ids for which the
metric is returned. Options are `[0]`, `[1]`, or `[0, 1]`. With
`[0]` (or `[1]`), the IoU metric for class 0 (or class 1,
respectively) is returned. With `[0, 1]`, the mean of IoUs for the
two classes is returned.

@param threshold
A threshold that applies to the prediction logits to convert
them to either predicted class 0 if the logit is below `threshold`
or predicted class 1 if the logit is above `threshold`.

@param name
(Optional) string name of the metric instance.

@param dtype
(Optional) data type of the metric result.

@param ...
Passed on to the Python callable

@export
@family metric
@seealso
+ <https://www.tensorflow.org/api_docs/python/tf/keras/metrics/BinaryIoU>
