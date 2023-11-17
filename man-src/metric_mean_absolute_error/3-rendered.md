Computes the mean absolute error between the labels and predictions.

@description

Formula:


```r
loss <- mean(abs(y_true - y_pred))
```

# Examples
Standalone usage:


```r
m <- metric_mean_absolute_error()
m$update_state(rbind(c(0, 1), c(0, 0)), rbind(c(1, 1), c(0, 0)))
m$result()
```

```
## tf.Tensor(0.25, shape=(), dtype=float32)
```

```r
m$reset_state()
m$update_state(rbind(c(0, 1), c(0, 0)), rbind(c(1, 1), c(0, 0)),
               sample_weight = c(1, 0))
m$result()
```

```
## tf.Tensor(0.5, shape=(), dtype=float32)
```

Usage with `compile()` API:


```r
model %>% compile(
    optimizer = 'sgd',
    loss = 'mse',
    metrics = list(metric_mean_absolute_error()))
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
@family losses
@family metrics
@family regression metrics
@seealso
+ <https:/keras.io/api/metrics/regression_metrics#meanabsoluteerror-class>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/metrics/MeanAbsoluteError>

