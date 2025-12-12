# Computes mean Intersection-Over-Union metric for one-hot encoded labels.

Formula:

    iou <- true_positives / (true_positives + false_positives + false_negatives)

Intersection-Over-Union is a common evaluation metric for semantic image
segmentation.

To compute IoUs, the predictions are accumulated in a confusion matrix,
weighted by `sample_weight` and the metric is then calculated from it.

If `sample_weight` is `NULL`, weights default to 1. Use `sample_weight`
of 0 to mask values.

This class can be used to compute the mean IoU for multi-class
classification tasks where the labels are one-hot encoded (the last axis
should have one dimension per class). Note that the predictions should
also have the same shape. To compute the mean IoU, first the labels and
predictions are converted back into integer format by taking the argmax
over the class axis. Then the same computation steps as for the base
`MeanIoU` class apply.

Note, if there is only one channel in the labels and predictions, this
class is the same as class `metric_mean_iou`. In this case, use
`metric_mean_iou` instead.

Also, make sure that `num_classes` is equal to the number of classes in
the data, to avoid a "labels out of bound" error when the confusion
matrix is computed.

## Usage

``` r
metric_one_hot_mean_iou(
  ...,
  num_classes,
  name = NULL,
  dtype = NULL,
  ignore_class = NULL,
  sparse_y_pred = FALSE,
  axis = -1L
)
```

## Arguments

- ...:

  For forward/backward compatability.

- num_classes:

  The possible number of labels the prediction task can have.

- name:

  (Optional) string name of the metric instance.

- dtype:

  (Optional) data type of the metric result.

- ignore_class:

  Optional integer. The ID of a class to be ignored during metric
  computation. This is useful, for example, in segmentation problems
  featuring a "void" class (commonly -1 or 255) in segmentation maps. By
  default (`ignore_class=NULL`), all classes are considered.

- sparse_y_pred:

  Whether predictions are encoded using natural numbers or probability
  distribution vectors. If `FALSE`, the `argmax` function will be used
  to determine each sample's most likely associated label.

- axis:

  (Optional) The dimension containing the logits. Defaults to `-1`.

## Value

a `Metric` instance is returned. The `Metric` instance can be passed
directly to `compile(metrics = )`, or used as a standalone object. See
[`?Metric`](https://keras3.posit.co/dev/reference/Metric.md) for example
usage.

## Examples

Standalone usage:

    y_true <- rbind(c(0, 0, 1), c(1, 0, 0), c(0, 1, 0), c(1, 0, 0))
    y_pred <- rbind(c(0.2, 0.3, 0.5), c(0.1, 0.2, 0.7), c(0.5, 0.3, 0.1),
                    c(0.1, 0.4, 0.5))
    sample_weight <- 10 * c(0.1, 0.2, 0.3, 0.4)
    m <- metric_one_hot_mean_iou(num_classes = 3L)
    m$update_state(
        y_true = y_true, y_pred = y_pred, sample_weight = sample_weight)

    ## tf.Tensor(
    ## [[0 0 6]
    ##  [3 0 0]
    ##  [0 0 1]], shape=(3, 3), dtype=int64)

    m$result()

    ## tf.Tensor(0.04761905, shape=(), dtype=float32)

Usage with
[`compile()`](https://generics.r-lib.org/reference/compile.html) API:

    model %>% compile(
        optimizer = 'sgd',
        loss = 'mse',
        metrics = list(metric_one_hot_mean_iou(num_classes = 3L)))

## See also

Other iou metrics:  
[`metric_binary_iou()`](https://keras3.posit.co/dev/reference/metric_binary_iou.md)  
[`metric_iou()`](https://keras3.posit.co/dev/reference/metric_iou.md)  
[`metric_mean_iou()`](https://keras3.posit.co/dev/reference/metric_mean_iou.md)  
[`metric_one_hot_iou()`](https://keras3.posit.co/dev/reference/metric_one_hot_iou.md)  

Other metrics:  
[`Metric()`](https://keras3.posit.co/dev/reference/Metric.md)  
[`custom_metric()`](https://keras3.posit.co/dev/reference/custom_metric.md)  
[`metric_auc()`](https://keras3.posit.co/dev/reference/metric_auc.md)  
[`metric_binary_accuracy()`](https://keras3.posit.co/dev/reference/metric_binary_accuracy.md)  
[`metric_binary_crossentropy()`](https://keras3.posit.co/dev/reference/metric_binary_crossentropy.md)  
[`metric_binary_focal_crossentropy()`](https://keras3.posit.co/dev/reference/metric_binary_focal_crossentropy.md)  
[`metric_binary_iou()`](https://keras3.posit.co/dev/reference/metric_binary_iou.md)  
[`metric_categorical_accuracy()`](https://keras3.posit.co/dev/reference/metric_categorical_accuracy.md)  
[`metric_categorical_crossentropy()`](https://keras3.posit.co/dev/reference/metric_categorical_crossentropy.md)  
[`metric_categorical_focal_crossentropy()`](https://keras3.posit.co/dev/reference/metric_categorical_focal_crossentropy.md)  
[`metric_categorical_hinge()`](https://keras3.posit.co/dev/reference/metric_categorical_hinge.md)  
[`metric_concordance_correlation()`](https://keras3.posit.co/dev/reference/metric_concordance_correlation.md)  
[`metric_cosine_similarity()`](https://keras3.posit.co/dev/reference/metric_cosine_similarity.md)  
[`metric_f1_score()`](https://keras3.posit.co/dev/reference/metric_f1_score.md)  
[`metric_false_negatives()`](https://keras3.posit.co/dev/reference/metric_false_negatives.md)  
[`metric_false_positives()`](https://keras3.posit.co/dev/reference/metric_false_positives.md)  
[`metric_fbeta_score()`](https://keras3.posit.co/dev/reference/metric_fbeta_score.md)  
[`metric_hinge()`](https://keras3.posit.co/dev/reference/metric_hinge.md)  
[`metric_huber()`](https://keras3.posit.co/dev/reference/metric_huber.md)  
[`metric_iou()`](https://keras3.posit.co/dev/reference/metric_iou.md)  
[`metric_kl_divergence()`](https://keras3.posit.co/dev/reference/metric_kl_divergence.md)  
[`metric_log_cosh()`](https://keras3.posit.co/dev/reference/metric_log_cosh.md)  
[`metric_log_cosh_error()`](https://keras3.posit.co/dev/reference/metric_log_cosh_error.md)  
[`metric_mean()`](https://keras3.posit.co/dev/reference/metric_mean.md)  
[`metric_mean_absolute_error()`](https://keras3.posit.co/dev/reference/metric_mean_absolute_error.md)  
[`metric_mean_absolute_percentage_error()`](https://keras3.posit.co/dev/reference/metric_mean_absolute_percentage_error.md)  
[`metric_mean_iou()`](https://keras3.posit.co/dev/reference/metric_mean_iou.md)  
[`metric_mean_squared_error()`](https://keras3.posit.co/dev/reference/metric_mean_squared_error.md)  
[`metric_mean_squared_logarithmic_error()`](https://keras3.posit.co/dev/reference/metric_mean_squared_logarithmic_error.md)  
[`metric_mean_wrapper()`](https://keras3.posit.co/dev/reference/metric_mean_wrapper.md)  
[`metric_one_hot_iou()`](https://keras3.posit.co/dev/reference/metric_one_hot_iou.md)  
[`metric_pearson_correlation()`](https://keras3.posit.co/dev/reference/metric_pearson_correlation.md)  
[`metric_poisson()`](https://keras3.posit.co/dev/reference/metric_poisson.md)  
[`metric_precision()`](https://keras3.posit.co/dev/reference/metric_precision.md)  
[`metric_precision_at_recall()`](https://keras3.posit.co/dev/reference/metric_precision_at_recall.md)  
[`metric_r2_score()`](https://keras3.posit.co/dev/reference/metric_r2_score.md)  
[`metric_recall()`](https://keras3.posit.co/dev/reference/metric_recall.md)  
[`metric_recall_at_precision()`](https://keras3.posit.co/dev/reference/metric_recall_at_precision.md)  
[`metric_root_mean_squared_error()`](https://keras3.posit.co/dev/reference/metric_root_mean_squared_error.md)  
[`metric_sensitivity_at_specificity()`](https://keras3.posit.co/dev/reference/metric_sensitivity_at_specificity.md)  
[`metric_sparse_categorical_accuracy()`](https://keras3.posit.co/dev/reference/metric_sparse_categorical_accuracy.md)  
[`metric_sparse_categorical_crossentropy()`](https://keras3.posit.co/dev/reference/metric_sparse_categorical_crossentropy.md)  
[`metric_sparse_top_k_categorical_accuracy()`](https://keras3.posit.co/dev/reference/metric_sparse_top_k_categorical_accuracy.md)  
[`metric_specificity_at_sensitivity()`](https://keras3.posit.co/dev/reference/metric_specificity_at_sensitivity.md)  
[`metric_squared_hinge()`](https://keras3.posit.co/dev/reference/metric_squared_hinge.md)  
[`metric_sum()`](https://keras3.posit.co/dev/reference/metric_sum.md)  
[`metric_top_k_categorical_accuracy()`](https://keras3.posit.co/dev/reference/metric_top_k_categorical_accuracy.md)  
[`metric_true_negatives()`](https://keras3.posit.co/dev/reference/metric_true_negatives.md)  
[`metric_true_positives()`](https://keras3.posit.co/dev/reference/metric_true_positives.md)  
