# Computes how often integer targets are in the top `K` predictions.

Computes how often integer targets are in the top `K` predictions.

By default, the arguments expected by `update_state()` are:

- `y_true`: a tensor of shape `(batch_size)` representing indices of
  true categories.

- `y_pred`: a tensor of shape `(batch_size, num_categories)` containing
  the scores for each sample for all possible categories.

With `from_sorted_ids=TRUE`, the arguments expected by `update_state`
are:

- `y_true`: a tensor of shape `(batch_size)` representing indices or IDs
  of true categories.

- `y_pred`: a tensor of shape `(batch_size, N)` containing the indices
  or IDs of the top `N` categories sorted in order from highest score to
  lowest score. `N` must be greater or equal to `k`.

The `from_sorted_ids=TRUE` option can be more efficient when the set of
categories is very large and the model has an optimized way to retrieve
the top ones either without scoring or without maintaining the scores
for all the possible categories.

## Usage

``` r
metric_sparse_top_k_categorical_accuracy(
  y_true,
  y_pred,
  k = 5L,
  ...,
  name = "sparse_top_k_categorical_accuracy",
  dtype = NULL,
  from_sorted_ids = FALSE
)
```

## Arguments

- y_true:

  Tensor of true targets.

- y_pred:

  Tensor of predicted targets.

- k:

  (Optional) Number of top elements to look at for computing accuracy.
  Defaults to `5`.

- ...:

  For forward/backward compatability.

- name:

  (Optional) string name of the metric instance.

- dtype:

  (Optional) data type of the metric result.

- from_sorted_ids:

  (Optional) When `FALSE`, the default, the tensor passed in `y_pred`
  contains the unsorted scores of all possible categories. When `TRUE`,
  `y_pred` contains the indices or IDs for the top categories.

## Value

If `y_true` and `y_pred` are missing, a `Metric` instance is returned.
The `Metric` instance that can be passed directly to
`compile(metrics = )`, or used as a standalone object. See
[`?Metric`](https://keras3.posit.co/reference/Metric.md) for example
usage. If `y_true` and `y_pred` are provided, then a tensor with the
computed value is returned.

## Usage

Standalone usage:

    m <- metric_sparse_top_k_categorical_accuracy(k = 1L)
    m$update_state(
      rbind(2, 1),
      op_array(rbind(c(0.1, 0.9, 0.8), c(0.05, 0.95, 0)), dtype = "float32")
    )
    m$result()

    ## tf.Tensor(0.5, shape=(), dtype=float32)

    m$reset_state()
    m$update_state(
      rbind(2, 1),
      op_array(rbind(c(0.1, 0.9, 0.8), c(0.05, 0.95, 0)), dtype = "float32"),
      sample_weight = c(0.7, 0.3)
    )
    m$result()

    ## tf.Tensor(0.3, shape=(), dtype=float32)

    m <- metric_sparse_top_k_categorical_accuracy(k = 1, from_sorted_ids = TRUE)
    m$update_state(array(c(2, 1)), rbind(c(1, 0, 3),
                                         c(1, 2, 3)))
    m$result()

    ## tf.Tensor(0.5, shape=(), dtype=float32)

Usage with
[`compile()`](https://generics.r-lib.org/reference/compile.html) API:

    model %>% compile(optimizer = 'sgd',
                      loss = 'sparse_categorical_crossentropy',
                      metrics = list(metric_sparse_top_k_categorical_accuracy()))

## See also

- <https://keras.io/api/metrics/accuracy_metrics#sparsetopkcategoricalaccuracy-class>

Other accuracy metrics:  
[`metric_binary_accuracy()`](https://keras3.posit.co/reference/metric_binary_accuracy.md)  
[`metric_categorical_accuracy()`](https://keras3.posit.co/reference/metric_categorical_accuracy.md)  
[`metric_sparse_categorical_accuracy()`](https://keras3.posit.co/reference/metric_sparse_categorical_accuracy.md)  
[`metric_top_k_categorical_accuracy()`](https://keras3.posit.co/reference/metric_top_k_categorical_accuracy.md)  

Other metrics:  
[`Metric()`](https://keras3.posit.co/reference/Metric.md)  
[`custom_metric()`](https://keras3.posit.co/reference/custom_metric.md)  
[`metric_auc()`](https://keras3.posit.co/reference/metric_auc.md)  
[`metric_binary_accuracy()`](https://keras3.posit.co/reference/metric_binary_accuracy.md)  
[`metric_binary_crossentropy()`](https://keras3.posit.co/reference/metric_binary_crossentropy.md)  
[`metric_binary_focal_crossentropy()`](https://keras3.posit.co/reference/metric_binary_focal_crossentropy.md)  
[`metric_binary_iou()`](https://keras3.posit.co/reference/metric_binary_iou.md)  
[`metric_categorical_accuracy()`](https://keras3.posit.co/reference/metric_categorical_accuracy.md)  
[`metric_categorical_crossentropy()`](https://keras3.posit.co/reference/metric_categorical_crossentropy.md)  
[`metric_categorical_focal_crossentropy()`](https://keras3.posit.co/reference/metric_categorical_focal_crossentropy.md)  
[`metric_categorical_hinge()`](https://keras3.posit.co/reference/metric_categorical_hinge.md)  
[`metric_concordance_correlation()`](https://keras3.posit.co/reference/metric_concordance_correlation.md)  
[`metric_cosine_similarity()`](https://keras3.posit.co/reference/metric_cosine_similarity.md)  
[`metric_f1_score()`](https://keras3.posit.co/reference/metric_f1_score.md)  
[`metric_false_negatives()`](https://keras3.posit.co/reference/metric_false_negatives.md)  
[`metric_false_positives()`](https://keras3.posit.co/reference/metric_false_positives.md)  
[`metric_fbeta_score()`](https://keras3.posit.co/reference/metric_fbeta_score.md)  
[`metric_hinge()`](https://keras3.posit.co/reference/metric_hinge.md)  
[`metric_huber()`](https://keras3.posit.co/reference/metric_huber.md)  
[`metric_iou()`](https://keras3.posit.co/reference/metric_iou.md)  
[`metric_kl_divergence()`](https://keras3.posit.co/reference/metric_kl_divergence.md)  
[`metric_log_cosh()`](https://keras3.posit.co/reference/metric_log_cosh.md)  
[`metric_log_cosh_error()`](https://keras3.posit.co/reference/metric_log_cosh_error.md)  
[`metric_mean()`](https://keras3.posit.co/reference/metric_mean.md)  
[`metric_mean_absolute_error()`](https://keras3.posit.co/reference/metric_mean_absolute_error.md)  
[`metric_mean_absolute_percentage_error()`](https://keras3.posit.co/reference/metric_mean_absolute_percentage_error.md)  
[`metric_mean_iou()`](https://keras3.posit.co/reference/metric_mean_iou.md)  
[`metric_mean_squared_error()`](https://keras3.posit.co/reference/metric_mean_squared_error.md)  
[`metric_mean_squared_logarithmic_error()`](https://keras3.posit.co/reference/metric_mean_squared_logarithmic_error.md)  
[`metric_mean_wrapper()`](https://keras3.posit.co/reference/metric_mean_wrapper.md)  
[`metric_one_hot_iou()`](https://keras3.posit.co/reference/metric_one_hot_iou.md)  
[`metric_one_hot_mean_iou()`](https://keras3.posit.co/reference/metric_one_hot_mean_iou.md)  
[`metric_pearson_correlation()`](https://keras3.posit.co/reference/metric_pearson_correlation.md)  
[`metric_poisson()`](https://keras3.posit.co/reference/metric_poisson.md)  
[`metric_precision()`](https://keras3.posit.co/reference/metric_precision.md)  
[`metric_precision_at_recall()`](https://keras3.posit.co/reference/metric_precision_at_recall.md)  
[`metric_r2_score()`](https://keras3.posit.co/reference/metric_r2_score.md)  
[`metric_recall()`](https://keras3.posit.co/reference/metric_recall.md)  
[`metric_recall_at_precision()`](https://keras3.posit.co/reference/metric_recall_at_precision.md)  
[`metric_root_mean_squared_error()`](https://keras3.posit.co/reference/metric_root_mean_squared_error.md)  
[`metric_sensitivity_at_specificity()`](https://keras3.posit.co/reference/metric_sensitivity_at_specificity.md)  
[`metric_sparse_categorical_accuracy()`](https://keras3.posit.co/reference/metric_sparse_categorical_accuracy.md)  
[`metric_sparse_categorical_crossentropy()`](https://keras3.posit.co/reference/metric_sparse_categorical_crossentropy.md)  
[`metric_specificity_at_sensitivity()`](https://keras3.posit.co/reference/metric_specificity_at_sensitivity.md)  
[`metric_squared_hinge()`](https://keras3.posit.co/reference/metric_squared_hinge.md)  
[`metric_sum()`](https://keras3.posit.co/reference/metric_sum.md)  
[`metric_top_k_categorical_accuracy()`](https://keras3.posit.co/reference/metric_top_k_categorical_accuracy.md)  
[`metric_true_negatives()`](https://keras3.posit.co/reference/metric_true_negatives.md)  
[`metric_true_positives()`](https://keras3.posit.co/reference/metric_true_positives.md)  
