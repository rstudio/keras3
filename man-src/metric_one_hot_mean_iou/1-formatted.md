Computes mean Intersection-Over-Union metric for one-hot encoded labels.

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

This class can be used to compute the mean IoU for multi-class
classification tasks where the labels are one-hot encoded (the last axis
should have one dimension per class). Note that the predictions should also
have the same shape. To compute the mean IoU, first the labels and
predictions are converted back into integer format by taking the argmax over
the class axis. Then the same computation steps as for the base `MeanIoU`
class apply.

Note, if there is only one channel in the labels and predictions, this class
is the same as class `MeanIoU`. In this case, use `MeanIoU` instead.

Also, make sure that `num_classes` is equal to the number of classes in the
data, to avoid a "labels out of bound" error when the confusion matrix is
computed.

# Examples
Standalone usage:

```python
y_true = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0], [1, 0, 0]])
y_pred = np.array([[0.2, 0.3, 0.5], [0.1, 0.2, 0.7], [0.5, 0.3, 0.1],
                      [0.1, 0.4, 0.5]])
sample_weight = [0.1, 0.2, 0.3, 0.4]
m = keras.metrics.OneHotMeanIoU(num_classes=3)
m.update_state(
    y_true=y_true, y_pred=y_pred, sample_weight=sample_weight)
# cm = [[0, 0, 0.2+0.4],
#       [0.3, 0, 0],
#       [0, 0, 0.1]]
# sum_row = [0.3, 0, 0.7], sum_col = [0.6, 0.3, 0.1]
# true_positives = [0, 0, 0.1]
# single_iou = true_positives / (sum_row + sum_col - true_positives))
# mean_iou = (0 + 0 + 0.1 / (0.7 + 0.1 - 0.1)) / 3
m.result()
# 0.048
```

Usage with `compile()` API:

```python
model.compile(
    optimizer='sgd',
    loss='mse',
    metrics=[keras.metrics.OneHotMeanIoU(num_classes=3)])
```

@param num_classes
The possible number of labels the prediction task can have.

@param name
(Optional) string name of the metric instance.

@param dtype
(Optional) data type of the metric result.

@param ignore_class
Optional integer. The ID of a class to be ignored during
metric computation. This is useful, for example, in segmentation
problems featuring a "void" class (commonly -1 or 255) in
segmentation maps. By default (`ignore_class=None`), all classes are
considered.

@param sparse_y_pred
Whether predictions are encoded using natural numbers or
probability distribution vectors. If `False`, the `argmax`
function will be used to determine each sample's most likely
associated label.

@param axis
(Optional) The dimension containing the logits. Defaults to `-1`.

@param ...
Passed on to the Python callable

@export
@family iou metrics
@family metrics
@seealso
+ <https://www.tensorflow.org/api_docs/python/tf/keras/metrics/OneHotMeanIoU>
