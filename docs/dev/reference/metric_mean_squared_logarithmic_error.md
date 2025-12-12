# Computes mean squared logarithmic error between `y_true` and `y_pred`.

Formula:

    loss <- mean(square(log(y_true + 1) - log(y_pred + 1)), axis=-1)

Note that `y_pred` and `y_true` cannot be less or equal to 0. Negative
values and 0 values will be replaced with `keras$backend$epsilon()`
(default to `1e-7`).

## Usage

``` r
metric_mean_squared_logarithmic_error(
  y_true,
  y_pred,
  ...,
  name = "mean_squared_logarithmic_error",
  dtype = NULL
)
```

## Arguments

- y_true:

  Ground truth values with shape = `[batch_size, d0, .. dN]`.

- y_pred:

  The predicted values with shape = `[batch_size, d0, .. dN]`.

- ...:

  For forward/backward compatability.

- name:

  (Optional) string name of the metric instance.

- dtype:

  (Optional) data type of the metric result.

## Value

If `y_true` and `y_pred` are missing, a `Metric` instance is returned.
The `Metric` instance that can be passed directly to
`compile(metrics = )`, or used as a standalone object. See
[`?Metric`](https://keras3.posit.co/dev/reference/Metric.md) for example
usage. If `y_true` and `y_pred` are provided, then a tensor with the
computed value is returned.

## Examples

Standalone usage:

    m <- metric_mean_squared_logarithmic_error()
    m$update_state(rbind(c(0, 1), c(0, 0)), rbind(c(1, 1), c(0, 0)))
    m$result()

    ## tf.Tensor(0.12011322, shape=(), dtype=float32)

    m$reset_state()
    m$update_state(rbind(c(0, 1), c(0, 0)), rbind(c(1, 1), c(0, 0)),
                   sample_weight = c(1, 0))
    m$result()

    ## tf.Tensor(0.24022643, shape=(), dtype=float32)

Usage with
[`compile()`](https://generics.r-lib.org/reference/compile.html) API:

    model %>% compile(
      optimizer = 'sgd',
      loss = 'mse',
      metrics = list(metric_mean_squared_logarithmic_error()))

## See also

- <https://keras.io/api/metrics/regression_metrics#meansquaredlogarithmicerror-class>

Other losses:  
[`Loss()`](https://keras3.posit.co/dev/reference/Loss.md)  
[`loss_binary_crossentropy()`](https://keras3.posit.co/dev/reference/loss_binary_crossentropy.md)  
[`loss_binary_focal_crossentropy()`](https://keras3.posit.co/dev/reference/loss_binary_focal_crossentropy.md)  
[`loss_categorical_crossentropy()`](https://keras3.posit.co/dev/reference/loss_categorical_crossentropy.md)  
[`loss_categorical_focal_crossentropy()`](https://keras3.posit.co/dev/reference/loss_categorical_focal_crossentropy.md)  
[`loss_categorical_generalized_cross_entropy()`](https://keras3.posit.co/dev/reference/loss_categorical_generalized_cross_entropy.md)  
[`loss_categorical_hinge()`](https://keras3.posit.co/dev/reference/loss_categorical_hinge.md)  
[`loss_circle()`](https://keras3.posit.co/dev/reference/loss_circle.md)  
[`loss_cosine_similarity()`](https://keras3.posit.co/dev/reference/loss_cosine_similarity.md)  
[`loss_ctc()`](https://keras3.posit.co/dev/reference/loss_ctc.md)  
[`loss_dice()`](https://keras3.posit.co/dev/reference/loss_dice.md)  
[`loss_hinge()`](https://keras3.posit.co/dev/reference/loss_hinge.md)  
[`loss_huber()`](https://keras3.posit.co/dev/reference/loss_huber.md)  
[`loss_kl_divergence()`](https://keras3.posit.co/dev/reference/loss_kl_divergence.md)  
[`loss_log_cosh()`](https://keras3.posit.co/dev/reference/loss_log_cosh.md)  
[`loss_mean_absolute_error()`](https://keras3.posit.co/dev/reference/loss_mean_absolute_error.md)  
[`loss_mean_absolute_percentage_error()`](https://keras3.posit.co/dev/reference/loss_mean_absolute_percentage_error.md)  
[`loss_mean_squared_error()`](https://keras3.posit.co/dev/reference/loss_mean_squared_error.md)  
[`loss_mean_squared_logarithmic_error()`](https://keras3.posit.co/dev/reference/loss_mean_squared_logarithmic_error.md)  
[`loss_poisson()`](https://keras3.posit.co/dev/reference/loss_poisson.md)  
[`loss_sparse_categorical_crossentropy()`](https://keras3.posit.co/dev/reference/loss_sparse_categorical_crossentropy.md)  
[`loss_squared_hinge()`](https://keras3.posit.co/dev/reference/loss_squared_hinge.md)  
[`loss_tversky()`](https://keras3.posit.co/dev/reference/loss_tversky.md)  
[`metric_binary_crossentropy()`](https://keras3.posit.co/dev/reference/metric_binary_crossentropy.md)  
[`metric_binary_focal_crossentropy()`](https://keras3.posit.co/dev/reference/metric_binary_focal_crossentropy.md)  
[`metric_categorical_crossentropy()`](https://keras3.posit.co/dev/reference/metric_categorical_crossentropy.md)  
[`metric_categorical_focal_crossentropy()`](https://keras3.posit.co/dev/reference/metric_categorical_focal_crossentropy.md)  
[`metric_categorical_hinge()`](https://keras3.posit.co/dev/reference/metric_categorical_hinge.md)  
[`metric_hinge()`](https://keras3.posit.co/dev/reference/metric_hinge.md)  
[`metric_huber()`](https://keras3.posit.co/dev/reference/metric_huber.md)  
[`metric_kl_divergence()`](https://keras3.posit.co/dev/reference/metric_kl_divergence.md)  
[`metric_log_cosh()`](https://keras3.posit.co/dev/reference/metric_log_cosh.md)  
[`metric_mean_absolute_error()`](https://keras3.posit.co/dev/reference/metric_mean_absolute_error.md)  
[`metric_mean_absolute_percentage_error()`](https://keras3.posit.co/dev/reference/metric_mean_absolute_percentage_error.md)  
[`metric_mean_squared_error()`](https://keras3.posit.co/dev/reference/metric_mean_squared_error.md)  
[`metric_poisson()`](https://keras3.posit.co/dev/reference/metric_poisson.md)  
[`metric_sparse_categorical_crossentropy()`](https://keras3.posit.co/dev/reference/metric_sparse_categorical_crossentropy.md)  
[`metric_squared_hinge()`](https://keras3.posit.co/dev/reference/metric_squared_hinge.md)  

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
[`metric_mean_wrapper()`](https://keras3.posit.co/dev/reference/metric_mean_wrapper.md)  
[`metric_one_hot_iou()`](https://keras3.posit.co/dev/reference/metric_one_hot_iou.md)  
[`metric_one_hot_mean_iou()`](https://keras3.posit.co/dev/reference/metric_one_hot_mean_iou.md)  
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

Other regression metrics:  
[`metric_concordance_correlation()`](https://keras3.posit.co/dev/reference/metric_concordance_correlation.md)  
[`metric_cosine_similarity()`](https://keras3.posit.co/dev/reference/metric_cosine_similarity.md)  
[`metric_log_cosh_error()`](https://keras3.posit.co/dev/reference/metric_log_cosh_error.md)  
[`metric_mean_absolute_error()`](https://keras3.posit.co/dev/reference/metric_mean_absolute_error.md)  
[`metric_mean_absolute_percentage_error()`](https://keras3.posit.co/dev/reference/metric_mean_absolute_percentage_error.md)  
[`metric_mean_squared_error()`](https://keras3.posit.co/dev/reference/metric_mean_squared_error.md)  
[`metric_pearson_correlation()`](https://keras3.posit.co/dev/reference/metric_pearson_correlation.md)  
[`metric_r2_score()`](https://keras3.posit.co/dev/reference/metric_r2_score.md)  
[`metric_root_mean_squared_error()`](https://keras3.posit.co/dev/reference/metric_root_mean_squared_error.md)  
