Compute the (weighted) sum of the given values.

@description
For example, if `values` is `[1, 3, 5, 7]` then their sum is 16.
If `sample_weight` was specified as `[1, 1, 0, 0]` then the sum would be 4.

This metric creates one variable, `total`.
This is ultimately returned as the sum value.

# Examples
```python
m = metrics.Sum()
m.update_state([1, 3, 5, 7])
m.result()
# 16.0
```

```python
m = metrics.Sum()
m.update_state([1, 3, 5, 7], sample_weight=[1, 1, 0, 0])
m.result()
# 4.0
```

@param name
(Optional) string name of the metric instance.

@param dtype
(Optional) data type of the metric result.

@param ...
Passed on to the Python callable

@export
@family metrics
@seealso
+ <https://www.tensorflow.org/api_docs/python/tf/keras/metrics/Sum>
