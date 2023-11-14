Calculates the number of false negatives.

@description
If `sample_weight` is given, calculates the sum of the weights of
false negatives. This metric creates one local variable, `accumulator`
that is used to keep track of the number of false negatives.

If `sample_weight` is `NULL`, weights default to 1.
Use `sample_weight` of 0 to mask values.

# Usage
Standalone usage:


```r
m <- metric_false_negatives()
m$update_state(c(0, 1, 1, 1), c(0, 1, 0, 0))
m$result()
```

```
## tf.Tensor(2.0, shape=(), dtype=float32)
```


```r
m$reset_state()
m$update_state(c(0, 1, 1, 1), c(0, 1, 0, 0), sample_weight=c(0, 0, 1, 0))
m$result()
```

```
## tf.Tensor(1.0, shape=(), dtype=float32)
```

```r
# 1.0
```

@param thresholds
(Optional) Defaults to `0.5`. A float value, or a Python
list of float threshold values in `[0, 1]`. A threshold is
compared with prediction values to determine the truth value of
predictions (i.e., above the threshold is `TRUE`, below is `FALSE`).
If used with a loss function that sets `from_logits=TRUE` (i.e. no
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
+ <https:/keras.io/api/metrics/classification_metrics#falsenegatives-class>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/metrics/FALSENegatives>

