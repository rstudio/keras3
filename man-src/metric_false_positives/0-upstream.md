keras.metrics.FalsePositives
__signature__
(thresholds=None, name=None, dtype=None)
__doc__
Calculates the number of false positives.

If `sample_weight` is given, calculates the sum of the weights of
false positives. This metric creates one local variable, `accumulator`
that is used to keep track of the number of false positives.

If `sample_weight` is `None`, weights default to 1.
Use `sample_weight` of 0 to mask values.

Args:
    thresholds: (Optional) Defaults to `0.5`. A float value, or a Python
        list/tuple of float threshold values in `[0, 1]`. A threshold is
        compared with prediction values to determine the truth value of
        predictions (i.e., above the threshold is `True`, below is `False`).
        If used with a loss function that sets `from_logits=True` (i.e. no
        sigmoid applied to predictions), `thresholds` should be set to 0.
        One metric value is generated for each threshold value.
    name: (Optional) string name of the metric instance.
    dtype: (Optional) data type of the metric result.

Usage:

Standalone usage:

>>> m = keras.metrics.FalsePositives()
>>> m.update_state([0, 1, 0, 0], [0, 0, 1, 1])
>>> m.result()
2.0

>>> m.reset_state()
>>> m.update_state([0, 1, 0, 0], [0, 0, 1, 1], sample_weight=[0, 0, 1, 0])
>>> m.result()
1.0
