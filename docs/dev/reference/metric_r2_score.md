# Computes R2 score.

Formula:

    sum_squares_residuals <- sum((y_true - y_pred) ** 2)
    sum_squares <- sum((y_true - mean(y_true)) ** 2)
    R2 <- 1 - sum_squares_residuals / sum_squares

This is also called the [coefficient of
determination](https://en.wikipedia.org/wiki/Coefficient_of_determination).

It indicates how close the fitted regression line is to ground-truth
data.

- The highest score possible is 1.0. It indicates that the predictors
  perfectly accounts for variation in the target.

- A score of 0.0 indicates that the predictors do not account for
  variation in the target.

- It can also be negative if the model is worse than random.

This metric can also compute the "Adjusted R2" score.

## Usage

``` r
metric_r2_score(
  ...,
  class_aggregation = "uniform_average",
  num_regressors = 0L,
  name = "r2_score",
  dtype = NULL
)
```

## Arguments

- ...:

  For forward/backward compatability.

- class_aggregation:

  Specifies how to aggregate scores corresponding to different output
  classes (or target dimensions), i.e. different dimensions on the last
  axis of the predictions. Equivalent to `multioutput` argument in
  Scikit-Learn. Should be one of `NULL` (no aggregation),
  `"uniform_average"`, `"variance_weighted_average"`.

- num_regressors:

  Number of independent regressors used ("Adjusted R2" score). 0 is the
  standard R2 score. Defaults to `0`.

- name:

  Optional. string name of the metric instance.

- dtype:

  Optional. data type of the metric result.

## Value

a `Metric` instance is returned. The `Metric` instance can be passed
directly to `compile(metrics = )`, or used as a standalone object. See
[`?Metric`](https://keras3.posit.co/dev/reference/Metric.md) for example
usage.

## Examples

    y_true <- rbind(1, 4, 3)
    y_pred <- rbind(2, 4, 4)
    metric <- metric_r2_score()
    metric$update_state(y_true, y_pred)
    metric$result()

    ## tf.Tensor(0.57142854, shape=(), dtype=float32)

## See also

Other regression metrics:  
[`metric_concordance_correlation()`](https://keras3.posit.co/dev/reference/metric_concordance_correlation.md)  
[`metric_cosine_similarity()`](https://keras3.posit.co/dev/reference/metric_cosine_similarity.md)  
[`metric_log_cosh_error()`](https://keras3.posit.co/dev/reference/metric_log_cosh_error.md)  
[`metric_mean_absolute_error()`](https://keras3.posit.co/dev/reference/metric_mean_absolute_error.md)  
[`metric_mean_absolute_percentage_error()`](https://keras3.posit.co/dev/reference/metric_mean_absolute_percentage_error.md)  
[`metric_mean_squared_error()`](https://keras3.posit.co/dev/reference/metric_mean_squared_error.md)  
[`metric_mean_squared_logarithmic_error()`](https://keras3.posit.co/dev/reference/metric_mean_squared_logarithmic_error.md)  
[`metric_pearson_correlation()`](https://keras3.posit.co/dev/reference/metric_pearson_correlation.md)  
[`metric_root_mean_squared_error()`](https://keras3.posit.co/dev/reference/metric_root_mean_squared_error.md)  

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
[`metric_one_hot_mean_iou()`](https://keras3.posit.co/dev/reference/metric_one_hot_mean_iou.md)  
[`metric_pearson_correlation()`](https://keras3.posit.co/dev/reference/metric_pearson_correlation.md)  
[`metric_poisson()`](https://keras3.posit.co/dev/reference/metric_poisson.md)  
[`metric_precision()`](https://keras3.posit.co/dev/reference/metric_precision.md)  
[`metric_precision_at_recall()`](https://keras3.posit.co/dev/reference/metric_precision_at_recall.md)  
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
