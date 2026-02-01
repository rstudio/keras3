# Calculates the Pearson Correlation Coefficient (PCC).

Formula:

    loss = mean(l2norm(y_true - mean(y_true) * l2norm(y_pred - mean(y_pred)))

PCC measures the linear relationship between the true values (`y_true`)
and the predicted values (`y_pred`). The coefficient ranges from -1 to
1, where a value of 1 implies a perfect positive linear correlation, 0
indicates no linear correlation, and -1 indicates a perfect negative
linear correlation.

This metric is widely used in regression tasks where the strength of the
linear relationship between predictions and true labels is an important
evaluation criterion.

## Usage

``` r
metric_pearson_correlation(
  y_true,
  y_pred,
  axis = -1L,
  ...,
  name = "pearson_correlation",
  dtype = NULL
)
```

## Arguments

- y_true:

  Tensor of true targets.

- y_pred:

  Tensor of predicted targets.

- axis:

  (Optional) integer or tuple of integers of the axis/axes along which
  to compute the metric. Defaults to `-1`.

- ...:

  For forward/backward compatability.

- name:

  (Optional) string name of the metric instance.

- dtype:

  (Optional) data type of the metric result.

## Examples

    pcc <- metric_pearson_correlation(axis = -1)
    y_true <- rbind(c(0, 1, 0.5),
                    c(1, 1, 0.2))
    y_pred <- rbind(c(0.1, 0.9, 0.5),
                    c(1, 0.9, 0.2))
    pcc$update_state(y_true, y_pred)
    pcc$result()

    ## tf.Tensor(0.99669963, shape=(), dtype=float32)

    # equivalent operation using R's stats::cor()
    mean(sapply(1:nrow(y_true), function(i) {
      cor(y_true[i, ], y_pred[i, ])
    }))

    ## [1] 0.9966996

Usage with
[`compile()`](https://generics.r-lib.org/reference/compile.html) API:

    model |> compile(
      optimizer = 'sgd',
      loss = 'mean_squared_error',
      metrics = c(keras.metrics.PearsonCorrelation())
    )

## See also

Other regression metrics:  
[`metric_concordance_correlation()`](https://keras3.posit.co/reference/metric_concordance_correlation.md)  
[`metric_cosine_similarity()`](https://keras3.posit.co/reference/metric_cosine_similarity.md)  
[`metric_log_cosh_error()`](https://keras3.posit.co/reference/metric_log_cosh_error.md)  
[`metric_mean_absolute_error()`](https://keras3.posit.co/reference/metric_mean_absolute_error.md)  
[`metric_mean_absolute_percentage_error()`](https://keras3.posit.co/reference/metric_mean_absolute_percentage_error.md)  
[`metric_mean_squared_error()`](https://keras3.posit.co/reference/metric_mean_squared_error.md)  
[`metric_mean_squared_logarithmic_error()`](https://keras3.posit.co/reference/metric_mean_squared_logarithmic_error.md)  
[`metric_r2_score()`](https://keras3.posit.co/reference/metric_r2_score.md)  
[`metric_root_mean_squared_error()`](https://keras3.posit.co/reference/metric_root_mean_squared_error.md)  

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
