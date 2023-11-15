Computes how often integer targets are in the top `K` predictions.

@description

# Usage
Standalone usage:


```r
m <- metric_sparse_top_k_categorical_accuracy(k = 1L)
m$update_state(
  rbind(2, 1),
  k_array(rbind(c(0.1, 0.9, 0.8), c(0.05, 0.95, 0)), dtype = "float32")
)
m$result()
```

```
## tf.Tensor(0.5, shape=(), dtype=float32)
```


```r
m$reset_state()
m$update_state(
  rbind(2, 1),
  k_array(rbind(c(0.1, 0.9, 0.8), c(0.05, 0.95, 0)), dtype = "float32"),
  sample_weight = c(0.7, 0.3)
)
m$result()
```

```
## tf.Tensor(0.3, shape=(), dtype=float32)
```

Usage with `compile()` API:


```r
model %>% compile(optimizer = 'sgd',
                  loss = 'sparse_categorical_crossentropy',
                  metrics = list(metric_sparse_top_k_categorical_accuracy()))
```

@param k
(Optional) Number of top elements to look at for computing accuracy.
Defaults to `5`.

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
+ <https:/keras.io/api/metrics/accuracy_metrics#sparsetopkcategoricalaccuracy-class>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/metrics/SparseTopKCategoricalAccuracy>

