keras.metrics.CategoricalHinge
__signature__
(name='categorical_hinge', dtype=None)
__doc__
Computes the categorical hinge metric between `y_true` and `y_pred`.

Args:
    name: (Optional) string name of the metric instance.
    dtype: (Optional) data type of the metric result.

Usage:

Standalone usage:
>>> m = keras.metrics.CategoricalHinge()
>>> m.update_state([[0, 1], [0, 0]], [[0.6, 0.4], [0.4, 0.6]])
>>> m.result().numpy()
1.4000001
>>> m.reset_state()
>>> m.update_state([[0, 1], [0, 0]], [[0.6, 0.4], [0.4, 0.6]],
...                sample_weight=[1, 0])
>>> m.result()
1.2
