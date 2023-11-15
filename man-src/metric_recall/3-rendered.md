Computes the recall of the predictions with respect to the labels.

@description
This metric creates two local variables, `true_positives` and
`false_negatives`, that are used to compute the recall. This value is
ultimately returned as `recall`, an idempotent operation that simply divides
`true_positives` by the sum of `true_positives` and `false_negatives`.

If `sample_weight` is `NULL`, weights default to 1.
Use `sample_weight` of 0 to mask values.

If `top_k` is set, recall will be computed as how often on average a class
among the labels of a batch entry is in the top-k predictions.

If `class_id` is specified, we calculate recall by considering only the
entries in the batch for which `class_id` is in the label, and computing the
fraction of them for which `class_id` is above the threshold and/or in the
top-k predictions.

# Usage
Standalone usage:


```r
m <- metric_recall()
m$update_state(c(0, 1, 1, 1), c(1, 0, 1, 1))
m$result()
```

```
## tf.Tensor(0.6666667, shape=(), dtype=float32)
```


```r
m$reset_state()
m$update_state(c(0, 1, 1, 1), c(1, 0, 1, 1), sample_weight = c(0, 0, 1, 0))
m$result()
```

```
## tf.Tensor(0.9999999, shape=(), dtype=float32)
```

Usage with `compile()` API:


```r
model %>% compile(optimizer = 'sgd',
                  loss = 'mse',
                  metrics = list(metric_recall()))
```

Usage with a loss with `from_logits=TRUE`:


```r
model %>% compile(optimizer = 'adam',
                  loss = loss_binary_crossentropy(from_logits=TRUE),
                  metrics = list(metric_recall(thresholds=0)))
```

@param thresholds
(Optional) A float value, or a Python list of float
threshold values in `[0, 1]`. A threshold is compared with
prediction values to determine the truth value of predictions (i.e.,
above the threshold is `TRUE`, below is `FALSE`). If used with a
loss function that sets `from_logits=TRUE` (i.e. no sigmoid
applied to predictions), `thresholds` should be set to 0.
One metric value is generated for each threshold value.
If neither `thresholds` nor `top_k` are set,
the default is to calculate recall with `thresholds=0.5`.

@param top_k
(Optional) Unset by default. An int value specifying the top-k
predictions to consider when calculating recall.

@param class_id
(Optional) Integer class ID for which we want binary metrics.
This must be in the half-open interval `[0, num_classes)`, where
`num_classes` is the last dimension of predictions.

@param name
(Optional) string name of the metric instance.

@param dtype
(Optional) data type of the metric result.

@param ...
Passed on to the Python callable

@export
@family confusion metrics
@family metrics
@seealso
+ <https:/keras.io/api/metrics/classification_metrics#recall-class>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/metrics/Recall>

