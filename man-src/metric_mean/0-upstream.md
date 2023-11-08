keras.metrics.Mean
__signature__
(name='mean', dtype=None)
__doc__
Compute the (weighted) mean of the given values.

For example, if values is `[1, 3, 5, 7]` then the mean is 4.
If `sample_weight` was specified as `[1, 1, 0, 0]` then the mean would be 2.

This metric creates two variables, `total` and `count`.
The mean value returned is simply `total` divided by `count`.

Args:
    name: (Optional) string name of the metric instance.
    dtype: (Optional) data type of the metric result.

Example:

>>> m = Mean()
>>> m.update_state([1, 3, 5, 7])
>>> m.result()
4.0

>>> m.reset_state()
>>> m.update_state([1, 3, 5, 7], sample_weight=[1, 1, 0, 0])
>>> m.result()
2.0
```
