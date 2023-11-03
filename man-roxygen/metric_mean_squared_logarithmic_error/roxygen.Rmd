Computes mean squared logarithmic error between `y_true` and `y_pred`.

@description
Formula:

```python
loss = mean(square(log(y_true + 1) - log(y_pred + 1)), axis=-1)
```

Note that `y_pred` and `y_true` cannot be less or equal to 0. Negative
values and 0 values will be replaced with `keras.backend.epsilon()`
(default to `1e-7`).
Formula:

```python
loss = mean(square(log(y_true + 1) - log(y_pred + 1)))
```

# Examples
```python
y_true = np.random.randint(0, 2, size=(2, 3))
y_pred = np.random.random(size=(2, 3))
loss = keras.losses.mean_squared_logarithmic_error(y_true, y_pred)
```
Standalone usage:

```python
m = keras.metrics.MeanSquaredLogarithmicError()
m.update_state([[0, 1], [0, 0]], [[1, 1], [0, 0]])
m.result()
# 0.12011322
m.reset_state()
m.update_state([[0, 1], [0, 0]], [[1, 1], [0, 0]],
               sample_weight=[1, 0])
m.result()
# 0.24022643
```

Usage with `compile()` API:

```python
model.compile(
    optimizer='sgd',
    loss='mse',
    metrics=[keras.metrics.MeanSquaredLogarithmicError()])
```

# Returns
Mean squared logarithmic error values with shape = `[batch_size, d0, ..
dN-1]`.

@param name (Optional) string name of the metric instance.
@param dtype (Optional) data type of the metric result.
@param y_true Ground truth values with shape = `[batch_size, d0, .. dN]`.
@param y_pred The predicted values with shape = `[batch_size, d0, .. dN]`.
@param ... Passed on to the Python callable

@export
@family metric
@seealso
+ <https://www.tensorflow.org/api_docs/python/tf/keras/metrics/MeanSquaredLogarithmicError>
