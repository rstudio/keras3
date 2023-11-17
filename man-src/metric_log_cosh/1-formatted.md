Logarithm of the hyperbolic cosine of the prediction error.

@description
Formula:
```python
loss = mean(log(cosh(y_pred - y_true)), axis=-1)
```

Note that `log(cosh(x))` is approximately equal to `(x ** 2) / 2` for small
`x` and to `abs(x) - log(2)` for large `x`. This means that 'logcosh' works
mostly like the mean squared error, but will not be so strongly affected by
the occasional wildly incorrect prediction.

# Examples
```python
y_true = [[0., 1.], [0., 0.]]
y_pred = [[1., 1.], [0., 0.]]
loss = keras.losses.log_cosh(y_true, y_pred)
# 0.108
```

@returns
    Logcosh error values with shape = `[batch_size, d0, .. dN-1]`.

@param y_true
Ground truth values with shape = `[batch_size, d0, .. dN]`.

@param y_pred
The predicted values with shape = `[batch_size, d0, .. dN]`.

@export
@family losses
@family metrics
@seealso
+ <https://www.tensorflow.org/api_docs/python/tf/keras/metrics/log_cosh>
