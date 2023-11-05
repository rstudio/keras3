Computes the hinge metric between `y_true` and `y_pred`.

`y_true` values are expected to be -1 or 1. If binary (0 or 1) labels are
provided we will convert them to -1 or 1.

Args:
    name: (Optional) string name of the metric instance.
    dtype: (Optional) data type of the metric result.

Usage:

Standalone usage:

>>> m = keras.metrics.SquaredHinge()
>>> m.update_state([[0, 1], [0, 0]], [[0.6, 0.4], [0.4, 0.6]])
>>> m.result()
1.86
>>> m.reset_state()
>>> m.update_state([[0, 1], [0, 0]], [[0.6, 0.4], [0.4, 0.6]],
...                sample_weight=[1, 0])
>>> m.result()
1.46
