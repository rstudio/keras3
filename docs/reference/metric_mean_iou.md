# Computes the mean Intersection-Over-Union metric.

Formula:

    iou <- true_positives / (true_positives + false_positives + false_negatives)

Intersection-Over-Union is a common evaluation metric for semantic image
segmentation.

To compute IoUs, the predictions are accumulated in a confusion matrix,
weighted by `sample_weight` and the metric is then calculated from it.

If `sample_weight` is `NULL`, weights default to 1. Use `sample_weight`
of 0 to mask values.

Note that this class first computes IoUs for all individual classes,
then returns the mean of these values.

## Usage

``` r
metric_mean_iou(
  ...,
  num_classes,
  name = NULL,
  dtype = NULL,
  ignore_class = NULL,
  sparse_y_true = TRUE,
  sparse_y_pred = TRUE,
  axis = -1L
)
```

## Arguments

- ...:

  For forward/backward compatability.

- num_classes:

  The possible number of labels the prediction task can have. This value
  must be provided, since a confusion matrix of dimension =
  `[num_classes, num_classes]` will be allocated.

- name:

  (Optional) string name of the metric instance.

- dtype:

  (Optional) data type of the metric result.

- ignore_class:

  Optional integer. The ID of a class to be ignored during metric
  computation. This is useful, for example, in segmentation problems
  featuring a "void" class (commonly -1 or 255) in segmentation maps. By
  default (`ignore_class=NULL`), all classes are considered.

- sparse_y_true:

  Whether labels are encoded using integers or dense floating point
  vectors. If `FALSE`, the `argmax` function is used to determine each
  sample's most likely associated label.

- sparse_y_pred:

  Whether predictions are encoded using integers or dense floating point
  vectors. If `FALSE`, the `argmax` function is used to determine each
  sample's most likely associated label.

- axis:

  (Optional) The dimension containing the logits. Defaults to `-1`.

## Value

a `Metric` instance is returned. The `Metric` instance can be passed
directly to `compile(metrics = )`, or used as a standalone object. See
[`?Metric`](https://keras3.posit.co/reference/Metric.md) for example
usage.

## Examples

Standalone usage:

    # cm = [[1, 1],
    #       [1, 1]]
    # sum_row = [2, 2], sum_col = [2, 2], true_positives = [1, 1]
    # iou = true_positives / (sum_row + sum_col - true_positives))
    # result = (1 / (2 + 2 - 1) + 1 / (2 + 2 - 1)) / 2 = 0.33
    m <- metric_mean_iou(num_classes = 2)
    m$update_state(c(0, 0, 1, 1), c(0, 1, 0, 1))

    ## tf.Tensor(
    ## [[1 1]
    ##  [1 1]], shape=(2, 2), dtype=int64)

    m$result()

    ## tf.Tensor(0.33333334, shape=(), dtype=float32)

    m$reset_state()
    m$update_state(c(0, 0, 1, 1), c(0, 1, 0, 1),
                   sample_weight= 10 * c(0.3, 0.3, 0.3, 0.1))

    ## tf.Tensor(
    ## [[3 3]
    ##  [3 1]], shape=(2, 2), dtype=int64)

    m$result()

    ## tf.Tensor(0.23809525, shape=(), dtype=float32)

Usage with
[`compile()`](https://generics.r-lib.org/reference/compile.html) API:

    model %>% compile(
      optimizer = 'sgd',
      loss = 'mse',
      metrics = list(metric_mean_iou(num_classes=2)))

## See also

- <https://keras.io/api/metrics/segmentation_metrics#meaniou-class>

Other iou metrics:  
[`metric_binary_iou()`](https://keras3.posit.co/reference/metric_binary_iou.md)  
[`metric_iou()`](https://keras3.posit.co/reference/metric_iou.md)  
[`metric_one_hot_iou()`](https://keras3.posit.co/reference/metric_one_hot_iou.md)  
[`metric_one_hot_mean_iou()`](https://keras3.posit.co/reference/metric_one_hot_mean_iou.md)  

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
