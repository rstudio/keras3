keras.metrics.Sum
__signature__
(name='sum', dtype=None)
__doc__
Compute the (weighted) sum of the given values.

For example, if `values` is `[1, 3, 5, 7]` then their sum is 16.
If `sample_weight` was specified as `[1, 1, 0, 0]` then the sum would be 4.

This metric creates one variable, `total`.
This is ultimately returned as the sum value.

Args:
    name: (Optional) string name of the metric instance.
    dtype: (Optional) data type of the metric result.

Example:

>>> m = metrics.Sum()
>>> m.update_state([1, 3, 5, 7])
>>> m.result()
16.0

>>> m = metrics.Sum()
>>> m.update_state([1, 3, 5, 7], sample_weight=[1, 1, 0, 0])
>>> m.result()
4.0
