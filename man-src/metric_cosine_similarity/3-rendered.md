Computes the cosine similarity between the labels and predictions.

@description
Formula:


```r
loss <- sum(l2_norm(y_true) * l2_norm(y_pred))
```
See: [Cosine Similarity](https://en.wikipedia.org/wiki/Cosine_similarity).
This metric keeps the average cosine similarity between `predictions` and
`labels` over a stream of data.

# Examples
Standalone usage:


```r
m <- metric_cosine_similarity(axis=2)
m$update_state(rbind(c(0., 1.), c(1., 1.)), rbind(c(1., 0.), c(1., 1.)))
m$result()
```

```
## tf.Tensor(0.5, shape=(), dtype=float32)
```

```r
m$reset_state()
m$update_state(rbind(c(0., 1.), c(1., 1.)), rbind(c(1., 0.), c(1., 1.)),
               sample_weight = c(0.3, 0.7))
m$result()
```

```
## tf.Tensor(0.7, shape=(), dtype=float32)
```

Usage with `compile()` API:


```r
model %>% compile(
  optimizer = 'sgd',
  loss = 'mse',
  metrics = list(metric_cosine_similarity(axis=2)))
```

@param name
(Optional) string name of the metric instance.

@param dtype
(Optional) data type of the metric result.

@param axis
(Optional) Defaults to `-1`. The dimension along which the cosine
similarity is computed.

@param ...
Passed on to the Python callable

@export
@family regression metrics
@family metrics
@seealso
+ <https:/keras.io/api/metrics/regression_metrics#cosinesimilarity-class>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/metrics/CosineSimilarity>

