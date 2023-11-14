Computes mean absolute percentage error between `y_true` and `y_pred`.

@description
Formula:

```python
loss = 100 * mean(abs((y_true - y_pred) / y_true), axis=-1)
```

Division by zero is prevented by dividing by `maximum(y_true, epsilon)`
where `epsilon = keras.backend.epsilon()`
(default to `1e-7`).

Formula:

```python
loss = 100 * mean(abs((y_true - y_pred) / y_true))
```

# Examples
Standalone usage:

```python
m = keras.metrics.MeanAbsolutePercentageError()
m.update_state([[0, 1], [0, 0]], [[1, 1], [0, 0]])
m.result()
# 250000000.0
m.reset_state()
m.update_state([[0, 1], [0, 0]], [[1, 1], [0, 0]],
               sample_weight=[1, 0])
m.result()
# 500000000.0
```

Usage with `compile()` API:

```python
model.compile(
    optimizer='sgd',
    loss='mse',
    metrics=[keras.metrics.MeanAbsolutePercentageError()])
```

@param name
(Optional) string name of the metric instance.

@param dtype
(Optional) data type of the metric result.

@param y_true
Ground truth values with shape = `[batch_size, d0, .. dN]`.

@param y_pred
The predicted values with shape = `[batch_size, d0, .. dN]`.

@param ...
Passed on to the Python callable

@export
@family regression metrics
@family metrics
@seealso
+ <https:/keras.io/api/metrics/regression_metrics#meanabsolutepercentageerror-class>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/metrics/MeanAbsolutePercentageError>
