Computes mean squared logarithmic error between `y_true` and `y_pred`.

@description
Formula:


```r
loss <- mean(square(log(y_true + 1) - log(y_pred + 1)), axis=-1)
```

Note that `y_pred` and `y_true` cannot be less or equal to 0. Negative
values and 0 values will be replaced with `keras$backend$epsilon()`
(default to `1e-7`).

# Examples
Standalone usage:


```r
m <- metric_mean_squared_logarithmic_error()
m$update_state(rbind(c(0, 1), c(0, 0)), rbind(c(1, 1), c(0, 0)))
m$result()
```

```
## tf.Tensor(0.12011322, shape=(), dtype=float32)
```

```r
m$reset_state()
m$update_state(rbind(c(0, 1), c(0, 0)), rbind(c(1, 1), c(0, 0)),
               sample_weight = c(1, 0))
m$result()
```

```
## tf.Tensor(0.24022643, shape=(), dtype=float32)
```

Usage with `compile()` API:


```r
model %>% compile(
  optimizer = 'sgd',
  loss = 'mse',
  metrics = list(metric_mean_squared_logarithmic_error()))
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
+ <https:/keras.io/api/metrics/regression_metrics#meansquaredlogarithmicerror-class>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/metrics/MeanSquaredLogarithmicError>

