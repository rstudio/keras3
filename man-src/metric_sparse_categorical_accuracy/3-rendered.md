Calculates how often predictions match integer labels.

@description

```r
acc <- sample_weight %*% (y_true == which.max(y_pred))
```

You can provide logits of classes as `y_pred`, since argmax of
logits and probabilities are same.

This metric creates two local variables, `total` and `count` that are used
to compute the frequency with which `y_pred` matches `y_true`. This
frequency is ultimately returned as `sparse categorical accuracy`: an
idempotent operation that simply divides `total` by `count`.

If `sample_weight` is `NULL`, weights default to 1.
Use `sample_weight` of 0 to mask values.

# Usage
Standalone usage:


```r
m <- metric_sparse_categorical_accuracy()
m$update_state(rbind(2L, 1L), rbind(c(0.1, 0.6, 0.3), c(0.05, 0.95, 0)))
m$result()
```

```
## tf.Tensor(0.5, shape=(), dtype=float32)
```


```r
m$reset_state()
m$update_state(rbind(2L, 1L), rbind(c(0.1, 0.6, 0.3), c(0.05, 0.95, 0)),
               sample_weight = c(0.7, 0.3))
m$result()
```

```
## tf.Tensor(0.3, shape=(), dtype=float32)
```

Usage with `compile()` API:


```r
model %>% compile(optimizer = 'sgd',
                  loss = 'sparse_categorical_crossentropy',
                  metrics = list(metric_sparse_categorical_accuracy()))
```

@param name
(Optional) string name of the metric instance.

@param dtype
(Optional) data type of the metric result.

@param y_true
Tensor of true targets.

@param y_pred
Tensor of predicted targets.

@param ...
Passed on to the Python callable

@export
@family accuracy metrics
@family metrics
@seealso
+ <https:/keras.io/api/metrics/accuracy_metrics#sparsecategoricalaccuracy-class>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/metrics/SparseCategoricalAccuracy>

