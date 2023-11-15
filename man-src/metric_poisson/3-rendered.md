Computes the Poisson metric between `y_true` and `y_pred`.

@description

Formula:


```r
metric <- y_pred - y_true * log(y_pred)
```

# Examples
Standalone usage:


```r
m <- metric_poisson()
m$update_state(rbind(c(0, 1), c(0, 0)), rbind(c(1, 1), c(0, 0)))
m$result()
```

```
## tf.Tensor(0.49999997, shape=(), dtype=float32)
```


```r
m$reset_state()
m$update_state(rbind(c(0, 1), c(0, 0)), rbind(c(1, 1), c(0, 0)),
               sample_weight = c(1, 0))
m$result()
```

```
## tf.Tensor(0.99999994, shape=(), dtype=float32)
```

Usage with `compile()` API:


```r
model %>% compile(
  optimizer = 'sgd',
  loss = 'mse',
  metrics = list(metric_poisson())
)
```

@param name
(Optional) string name of the metric instance.

@param dtype
(Optional) data type of the metric result.

@param y_true
Ground truth values. shape = `[batch_size, d0, .. dN]`.

@param y_pred
The predicted values. shape = `[batch_size, d0, .. dN]`.

@param ...
Passed on to the Python callable

@export
@family probabilistic metrics
@family metrics
@seealso
+ <https:/keras.io/api/metrics/probabilistic_metrics#poisson-class>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/metrics/Poisson>

