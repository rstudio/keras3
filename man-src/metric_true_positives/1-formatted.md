Calculates the number of true positives.

@description
If `sample_weight` is given, calculates the sum of the weights of
true positives. This metric creates one local variable, `true_positives`
that is used to keep track of the number of true positives.

If `sample_weight` is `None`, weights default to 1.
Use `sample_weight` of 0 to mask values.

# Usage
Standalone usage:

```python
m = keras.metrics.TruePositives()
m.update_state([0, 1, 1, 1], [1, 0, 1, 1])
m.result()
# 2.0
```

```python
m.reset_state()
m.update_state([0, 1, 1, 1], [1, 0, 1, 1], sample_weight=[0, 0, 1, 0])
m.result()
# 1.0
```

@param thresholds
(Optional) Defaults to `0.5`. A float value, or a Python
list/tuple of float threshold values in `[0, 1]`. A threshold is
compared with prediction values to determine the truth value of
predictions (i.e., above the threshold is `True`, below is `False`).
If used with a loss function that sets `from_logits=True` (i.e. no
sigmoid applied to predictions), `thresholds` should be set to 0.
One metric value is generated for each threshold value.

@param name
(Optional) string name of the metric instance.

@param dtype
(Optional) data type of the metric result.

@param ...
Passed on to the Python callable

@export
@family confusion metrics
@family metrics
@seealso
+ <https:/keras.io/api/metrics/classification_metrics#truepositives-class>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/metrics/TruePositives>
