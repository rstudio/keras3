# Computes the Intersection-Over-Union metric for class 0 and/or 1.

Formula:

    iou <- true_positives / (true_positives + false_positives + false_negatives)

Intersection-Over-Union is a common evaluation metric for semantic image
segmentation.

To compute IoUs, the predictions are accumulated in a confusion matrix,
weighted by `sample_weight` and the metric is then calculated from it.

If `sample_weight` is `NULL`, weights default to 1. Use `sample_weight`
of 0 to mask values.

This class can be used to compute IoUs for a binary classification task
where the predictions are provided as logits. First a `threshold` is
applied to the predicted values such that those that are below the
`threshold` are converted to class 0 and those that are above the
`threshold` are converted to class 1.

IoUs for classes 0 and 1 are then computed, the mean of IoUs for the
classes that are specified by `target_class_ids` is returned.

## Usage

``` r
metric_binary_iou(
  ...,
  target_class_ids = list(0L, 1L),
  threshold = 0.5,
  name = NULL,
  dtype = NULL
)
```

## Arguments

- ...:

  For forward/backward compatability.

- target_class_ids:

  A list or list of target class ids for which the metric is returned.
  Options are `0`, `1`, or `c(0, 1)`. With `0` (or `1`), the IoU metric
  for class 0 (or class 1, respectively) is returned. With `c(0, 1)`,
  the mean of IoUs for the two classes is returned.

- threshold:

  A threshold that applies to the prediction logits to convert them to
  either predicted class 0 if the logit is below `threshold` or
  predicted class 1 if the logit is above `threshold`.

- name:

  (Optional) string name of the metric instance.

- dtype:

  (Optional) data type of the metric result.

## Value

a `Metric` instance is returned. The `Metric` instance can be passed
directly to `compile(metrics = )`, or used as a standalone object. See
[`?Metric`](https://keras3.posit.co/reference/Metric.md) for example
usage.

## Note

with `threshold=0`, this metric has the same behavior as `IoU`.

## Examples

Standalone usage:

    m <- metric_binary_iou(target_class_ids=c(0L, 1L), threshold = 0.3)
    m$update_state(c(0, 1, 0, 1), c(0.1, 0.2, 0.4, 0.7))

    ## tf.Tensor(
    ## [[1 1]
    ##  [1 1]], shape=(2, 2), dtype=int64)

    m$result()

    ## tf.Tensor(0.33333334, shape=(), dtype=float32)

    m$reset_state()
    m$update_state(c(0, 1, 0, 1), c(0.1, 0.2, 0.4, 0.7),
                   sample_weight = 10 * c(0.2, 0.3, 0.4, 0.1))

    ## tf.Tensor(
    ## [[2 4]
    ##  [3 1]], shape=(2, 2), dtype=int64)

    m$result()

    ## tf.Tensor(0.1736111, shape=(), dtype=float32)

Usage with
[`compile()`](https://generics.r-lib.org/reference/compile.html) API:

    model %>% compile(
        optimizer = 'sgd',
        loss = 'mse',
        metrics = list(metric_binary_iou(
            target_class_ids = 0L,
            threshold = 0.5
        ))
    )

## See also

Other iou metrics:  
[`metric_iou()`](https://keras3.posit.co/reference/metric_iou.md)  
[`metric_mean_iou()`](https://keras3.posit.co/reference/metric_mean_iou.md)  
[`metric_one_hot_iou()`](https://keras3.posit.co/reference/metric_one_hot_iou.md)  
[`metric_one_hot_mean_iou()`](https://keras3.posit.co/reference/metric_one_hot_mean_iou.md)  

Other metrics:  
[`Metric()`](https://keras3.posit.co/reference/Metric.md)  
[`custom_metric()`](https://keras3.posit.co/reference/custom_metric.md)  
[`metric_auc()`](https://keras3.posit.co/reference/metric_auc.md)  
[`metric_binary_accuracy()`](https://keras3.posit.co/reference/metric_binary_accuracy.md)  
[`metric_binary_crossentropy()`](https://keras3.posit.co/reference/metric_binary_crossentropy.md)  
[`metric_binary_focal_crossentropy()`](https://keras3.posit.co/reference/metric_binary_focal_crossentropy.md)  
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
[`metric_sparse_top_k_categorical_accuracy()`](https://keras3.posit.co/reference/metric_sparse_top_k_categorical_accuracy.md)  
[`metric_specificity_at_sensitivity()`](https://keras3.posit.co/reference/metric_specificity_at_sensitivity.md)  
[`metric_squared_hinge()`](https://keras3.posit.co/reference/metric_squared_hinge.md)  
[`metric_sum()`](https://keras3.posit.co/reference/metric_sum.md)  
[`metric_top_k_categorical_accuracy()`](https://keras3.posit.co/reference/metric_top_k_categorical_accuracy.md)  
[`metric_true_negatives()`](https://keras3.posit.co/reference/metric_true_negatives.md)  
[`metric_true_positives()`](https://keras3.posit.co/reference/metric_true_positives.md)  
