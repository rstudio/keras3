Computes best sensitivity where specificity is >= specified value.

@description
`Sensitivity` measures the proportion of actual positives that are correctly
identified as such `(tp / (tp + fn))`.
`Specificity` measures the proportion of actual negatives that are correctly
identified as such `(tn / (tn + fp))`.

This metric creates four local variables, `true_positives`,
`true_negatives`, `false_positives` and `false_negatives` that are used to
compute the sensitivity at the given specificity. The threshold for the
given specificity value is computed and used to evaluate the corresponding
sensitivity.

If `sample_weight` is `None`, weights default to 1.
Use `sample_weight` of 0 to mask values.

If `class_id` is specified, we calculate precision by considering only the
entries in the batch for which `class_id` is above the threshold
predictions, and computing the fraction of them for which `class_id` is
indeed a correct label.

For additional information about specificity and sensitivity, see
[the following](https://en.wikipedia.org/wiki/Sensitivity_and_specificity).

# Usage
Standalone usage:

```python
m = keras.metrics.SensitivityAtSpecificity(0.5)
m.update_state([0, 0, 0, 1, 1], [0, 0.3, 0.8, 0.3, 0.8])
m.result()
# 0.5
```

```python
m.reset_state()
m.update_state([0, 0, 0, 1, 1], [0, 0.3, 0.8, 0.3, 0.8],
               sample_weight=[1, 1, 2, 2, 1])
m.result()
# 0.333333
```

Usage with `compile()` API:

```python
model.compile(
    optimizer='sgd',
    loss='mse',
    metrics=[keras.metrics.SensitivityAtSpecificity()])
```

@param specificity A scalar value in range `[0, 1]`.
@param num_thresholds (Optional) Defaults to 200. The number of thresholds to
    use for matching the given specificity.
@param class_id (Optional) Integer class ID for which we want binary metrics.
    This must be in the half-open interval `[0, num_classes)`, where
    `num_classes` is the last dimension of predictions.
@param name (Optional) string name of the metric instance.
@param dtype (Optional) data type of the metric result.
@param ... Passed on to the Python callable

@export
@family metric
@seealso
+ <https:/keras.io/api/metrics/classification_metrics#sensitivityatspecificity-class>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/metrics/SensitivityAtSpecificity>
