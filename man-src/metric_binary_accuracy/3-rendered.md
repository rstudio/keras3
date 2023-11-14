Calculates how often predictions match binary labels.

@description
This metric creates two local variables, `total` and `count` that are used
to compute the frequency with which `y_pred` matches `y_true`. This
frequency is ultimately returned as `binary accuracy`: an idempotent
operation that simply divides `total` by `count`.

If `sample_weight` is `NULL`, weights default to 1.
Use `sample_weight` of 0 to mask values.

# Usage
Standalone usage:


```r
m <- metric_binary_accuracy()
m$update_state(rbind(1, 1, 0, 0), rbind(0.98, 1, 0, 0.6))
m$result()
```

```
## tf.Tensor(0.75, shape=(), dtype=float32)
```

```r
# 0.75
```


```r
m$reset_state()
m$update_state(rbind(1, 1, 0, 0), rbind(0.98, 1, 0, 0.6),
               sample_weight = c(1, 0, 0, 1))
m$result()
```

```
## tf.Tensor(0.5, shape=(), dtype=float32)
```

```r
# 0.5
```

Usage with `compile()` API:


```r
model %>% compile(optimizer='sgd',
                  loss='binary_crossentropy',
                  metrics=list(metric_binary_accuracy()))
```

@param name
(Optional) string name of the metric instance.

@param dtype
(Optional) data type of the metric result.

@param threshold
(Optional) Float representing the threshold for deciding
whether prediction values are 1 or 0.

@param y_true
Tensor of true targets.

@param y_pred
Tensor of predicted targets.

@param ...
Passed on to the Python callable

@export
@family metric
@seealso
+ <https:/keras.io/api/metrics/accuracy_metrics#binaryaccuracy-class>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/metrics/BinaryAccuracy>

