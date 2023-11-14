Computes the mean squared error between `y_true` and `y_pred`.

@description
Formula:

```python
loss = mean(square(y_true - y_pred), axis=-1)
```

Formula:

```python
loss = mean(square(y_true - y_pred))
```

# Examples
```python
m = keras.metrics.MeanSquaredError()
m.update_state([[0, 1], [0, 0]], [[1, 1], [0, 0]])
m.result()
# 0.25
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
+ <https:/keras.io/api/metrics/regression_metrics#meansquarederror-class>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/metrics/MeanSquaredError>
