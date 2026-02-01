# Compute the (weighted) mean of the given values.

For example, if values is `c(1, 3, 5, 7)` then the mean is 4. If
`sample_weight` was specified as `c(1, 1, 0, 0)` then the mean would be
2.

This metric creates two variables, `total` and `count`. The mean value
returned is simply `total` divided by `count`.

## Usage

``` r
metric_mean(..., name = "mean", dtype = NULL)
```

## Arguments

- ...:

  For forward/backward compatability.

- name:

  (Optional) string name of the metric instance.

- dtype:

  (Optional) data type of the metric result.

## Value

a `Metric` instance is returned. The `Metric` instance can be passed
directly to `compile(metrics = )`, or used as a standalone object. See
[`?Metric`](https://keras3.posit.co/reference/Metric.md) for example
usage.

## Examples

    m <- metric_mean()
    m$update_state(c(1, 3, 5, 7))
    m$result()

    ## tf.Tensor(4.0, shape=(), dtype=float32)

    # calling a metric directly is equivalent to calling
    # m$update_state(); m$result()
    m <- metric_mean()
    m(c(1, 3, 5, 7))

    ## tf.Tensor(4.0, shape=(), dtype=float32)

    m$reset_state()
    m$update_state(c(1, 3, 5, 7), sample_weight = c(1, 1, 0, 0))
    m$result()

    ## tf.Tensor(2.0, shape=(), dtype=float32)

## See also

Other reduction metrics:  
[`metric_mean_wrapper()`](https://keras3.posit.co/reference/metric_mean_wrapper.md)  
[`metric_sum()`](https://keras3.posit.co/reference/metric_sum.md)  

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
[`metric_sparse_top_k_categorical_accuracy()`](https://keras3.posit.co/reference/metric_sparse_top_k_categorical_accuracy.md)  
[`metric_specificity_at_sensitivity()`](https://keras3.posit.co/reference/metric_specificity_at_sensitivity.md)  
[`metric_squared_hinge()`](https://keras3.posit.co/reference/metric_squared_hinge.md)  
[`metric_sum()`](https://keras3.posit.co/reference/metric_sum.md)  
[`metric_top_k_categorical_accuracy()`](https://keras3.posit.co/reference/metric_top_k_categorical_accuracy.md)  
[`metric_true_negatives()`](https://keras3.posit.co/reference/metric_true_negatives.md)  
[`metric_true_positives()`](https://keras3.posit.co/reference/metric_true_positives.md)  
