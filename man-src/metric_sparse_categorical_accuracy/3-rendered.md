Calculates how often predictions match integer labels.

@description
```python
acc = np.dot(sample_weight, np.equal(y_true, np.argmax(y_pred, axis=1))
```

You can provide logits of classes as `y_pred`, since argmax of
logits and probabilities are same.

This metric creates two local variables, `total` and `count` that are used
to compute the frequency with which `y_pred` matches `y_true`. This
frequency is ultimately returned as `sparse categorical accuracy`: an
idempotent operation that simply divides `total` by `count`.

If `sample_weight` is `None`, weights default to 1.
Use `sample_weight` of 0 to mask values.

# Usage
Standalone usage:

```python
m = keras.metrics.SparseCategoricalAccuracy()
m.update_state([[2], [1]], [[0.1, 0.6, 0.3], [0.05, 0.95, 0]])
m.result()
# 0.5
```

```python
m.reset_state()
m.update_state([[2], [1]], [[0.1, 0.6, 0.3], [0.05, 0.95, 0]],
               sample_weight=[0.7, 0.3])
m.result()
# 0.3
```

Usage with `compile()` API:

```python
model.compile(optimizer='sgd',
              loss='sparse_categorical_crossentropy',
              metrics=[keras.metrics.SparseCategoricalAccuracy()])
```

@param name
(Optional) string name of the metric instance.

@param dtype
(Optional) data type of the metric result.

@param y_true
Tensor of true targets.

@param y_pred
Tensor of predicted targets.

@param ...
Passed on to the Python callable

@export
@family accuracy metrics
@family metrics
@seealso
+ <https:/keras.io/api/metrics/accuracy_metrics#sparsecategoricalaccuracy-class>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/metrics/SparseCategoricalAccuracy>
