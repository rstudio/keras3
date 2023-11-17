Computes F-1 Score.

@description
Formula:

```python
f1_score = 2 * (precision * recall) / (precision + recall)
```
This is the harmonic mean of precision and recall.
Its output range is `[0, 1]`. It works for both multi-class
and multi-label classification.

# Examples
```python
metric = keras.metrics.F1Score(threshold=0.5)
y_true = np.array([[1, 1, 1],
                   [1, 0, 0],
                   [1, 1, 0]], np.int32)
y_pred = np.array([[0.2, 0.6, 0.7],
                   [0.2, 0.6, 0.6],
                   [0.6, 0.8, 0.0]], np.float32)
metric.update_state(y_true, y_pred)
result = metric.result()
# array([0.5      , 0.8      , 0.6666667], dtype=float32)
```

@returns
F-1 Score: float.

@param average
Type of averaging to be performed on data.
Acceptable values are `None`, `"micro"`, `"macro"`
and `"weighted"`. Defaults to `None`.
If `None`, no averaging is performed and `result()` will return
the score for each class.
If `"micro"`, compute metrics globally by counting the total
true positives, false negatives and false positives.
If `"macro"`, compute metrics for each label,
and return their unweighted mean.
This does not take label imbalance into account.
If `"weighted"`, compute metrics for each label,
and return their average weighted by support
(the number of true instances for each label).
This alters `"macro"` to account for label imbalance.
It can result in an score that is not between precision and recall.

@param threshold
Elements of `y_pred` greater than `threshold` are
converted to be 1, and the rest 0. If `threshold` is
`None`, the argmax of `y_pred` is converted to 1, and the rest to 0.

@param name
Optional. String name of the metric instance.

@param dtype
Optional. Data type of the metric result.

@param ...
Passed on to the Python callable

@export
@family f score metrics
@family metrics
@seealso
+ <https://www.tensorflow.org/api_docs/python/tf/keras/metrics/F1Score>
