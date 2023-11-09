Computes the mean absolute error between the labels and predictions.

@description
```python
loss = mean(abs(y_true - y_pred), axis=-1)
```
Formula:

```python
loss = mean(abs(y_true - y_pred))
```

# Examples
```python
y_true = np.random.randint(0, 2, size=(2, 3))
y_pred = np.random.random(size=(2, 3))
loss = keras.losses.mean_absolute_error(y_true, y_pred)
```
Standalone usage:

```python
m = keras.metrics.MeanAbsoluteError()
m.update_state([[0, 1], [0, 0]], [[1, 1], [0, 0]])
m.result()
# 0.25
m.reset_state()
m.update_state([[0, 1], [0, 0]], [[1, 1], [0, 0]],
               sample_weight=[1, 0])
m.result()
# 0.5
```

Usage with `compile()` API:

```python
model.compile(
    optimizer='sgd',
    loss='mse',
    metrics=[keras.metrics.MeanAbsoluteError()])
```

@returns
Mean absolute error values with shape = `[batch_size, d0, .. dN-1]`.

@param name (Optional) string name of the metric instance.
@param dtype (Optional) data type of the metric result.
@param y_true Ground truth values with shape = `[batch_size, d0, .. dN]`.
@param y_pred The predicted values with shape = `[batch_size, d0, .. dN]`.
@param ... Passed on to the Python callable

@export
@family metric
@seealso
+ <https:/keras.io/api/metrics/regression_metrics#meanabsoluteerror-class>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/metrics/MeanAbsoluteError>
