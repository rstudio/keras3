Computes root mean squared error metric between `y_true` and `y_pred`.

@description
Formula:


```r
loss <- sqrt(mean((y_pred - y_true) ^ 2))
```

# Examples
Standalone usage:


```r
m <- metric_root_mean_squared_error()
m$update_state(rbind(c(0, 1), c(0, 0)), rbind(c(1, 1), c(0, 0)))
m$result()
```

```
## tf.Tensor(0.5, shape=(), dtype=float32)
```


```r
m$reset_state()
m$update_state(rbind(c(0, 1), c(0, 0)), rbind(c(1, 1), c(0, 0)),
               sample_weight = c(1, 0))
m$result()
```

```
## tf.Tensor(0.70710677, shape=(), dtype=float32)
```

Usage with `compile()` API:


```r
model %>% compile(
  optimizer = 'sgd',
  loss = 'mse',
  metrics = list(metric_root_mean_squared_error()))
```

@param name
(Optional) string name of the metric instance.

@param dtype
(Optional) data type of the metric result.

@param ...
Passed on to the Python callable

@export
@family regression metrics
@family metrics
@seealso
+ <https:/keras.io/api/metrics/regression_metrics#rootmeansquarederror-class>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/metrics/RootMeanSquaredError>

