# Computes F-Beta score.

Formula:

    b2 <- beta^2
    f_beta_score <- (1 + b2) * (precision * recall) / (precision * b2 + recall)

This is the weighted harmonic mean of precision and recall. Its output
range is `[0, 1]`. It works for both multi-class and multi-label
classification.

## Usage

``` r
metric_fbeta_score(
  ...,
  average = NULL,
  beta = 1,
  threshold = NULL,
  name = "fbeta_score",
  dtype = NULL
)
```

## Arguments

- ...:

  For forward/backward compatability.

- average:

  Type of averaging to be performed across per-class results in the
  multi-class case. Acceptable values are `NULL`, `"micro"`, `"macro"`
  and `"weighted"`. Defaults to `NULL`. If `NULL`, no averaging is
  performed and `result()` will return the score for each class. If
  `"micro"`, compute metrics globally by counting the total true
  positives, false negatives and false positives. If `"macro"`, compute
  metrics for each label, and return their unweighted mean. This does
  not take label imbalance into account. If `"weighted"`, compute
  metrics for each label, and return their average weighted by support
  (the number of true instances for each label). This alters `"macro"`
  to account for label imbalance. It can result in an score that is not
  between precision and recall.

- beta:

  Determines the weight of given to recall in the harmonic mean between
  precision and recall (see pseudocode equation above). Defaults to `1`.

- threshold:

  Elements of `y_pred` greater than `threshold` are converted to be 1,
  and the rest 0. If `threshold` is `NULL`, the argmax of `y_pred` is
  converted to 1, and the rest to 0.

- name:

  Optional. String name of the metric instance.

- dtype:

  Optional. Data type of the metric result.

## Value

a `Metric` instance is returned. The `Metric` instance can be passed
directly to `compile(metrics = )`, or used as a standalone object. See
[`?Metric`](https://keras3.posit.co/reference/Metric.md) for example
usage.

## Examples

    metric <- metric_fbeta_score(beta = 2.0, threshold = 0.5)
    y_true <- rbind(c(1, 1, 1),
                    c(1, 0, 0),
                    c(1, 1, 0))
    y_pred <- rbind(c(0.2, 0.6, 0.7),
                    c(0.2, 0.6, 0.6),
                    c(0.6, 0.8, 0.0))
    metric$update_state(y_true, y_pred)
    metric$result()

    ## tf.Tensor([0.3846154  0.90909094 0.8333332 ], shape=(3), dtype=float32)

## Returns

F-Beta Score: float.

## See also

Other f score metrics:  
[`metric_f1_score()`](https://keras3.posit.co/reference/metric_f1_score.md)  

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
