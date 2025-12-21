# Logarithm of the hyperbolic cosine of the prediction error.

Formula:

    loss <- mean(log(cosh(y_pred - y_true)), axis=-1)

Note that `log(cosh(x))` is approximately equal to `(x ** 2) / 2` for
small `x` and to `abs(x) - log(2)` for large `x`. This means that
'logcosh' works mostly like the mean squared error, but will not be so
strongly affected by the occasional wildly incorrect prediction.

## Usage

``` r
metric_log_cosh(y_true, y_pred)
```

## Arguments

- y_true:

  Ground truth values with shape = `[batch_size, d0, .. dN]`.

- y_pred:

  The predicted values with shape = `[batch_size, d0, .. dN]`.

## Value

Logcosh error values with shape = `[batch_size, d0, .. dN-1]`.

## Examples

    y_true <- rbind(c(0., 1.), c(0., 0.))
    y_pred <- rbind(c(1., 1.), c(0., 0.))
    loss <- metric_log_cosh(y_true, y_pred)
    loss

    ## tf.Tensor([0.2168904 0.       ], shape=(2), dtype=float32)

## See also

Other losses:  
[`Loss()`](https://keras3.posit.co/reference/Loss.md)  
[`loss_binary_crossentropy()`](https://keras3.posit.co/reference/loss_binary_crossentropy.md)  
[`loss_binary_focal_crossentropy()`](https://keras3.posit.co/reference/loss_binary_focal_crossentropy.md)  
[`loss_categorical_crossentropy()`](https://keras3.posit.co/reference/loss_categorical_crossentropy.md)  
[`loss_categorical_focal_crossentropy()`](https://keras3.posit.co/reference/loss_categorical_focal_crossentropy.md)  
[`loss_categorical_generalized_cross_entropy()`](https://keras3.posit.co/reference/loss_categorical_generalized_cross_entropy.md)  
[`loss_categorical_hinge()`](https://keras3.posit.co/reference/loss_categorical_hinge.md)  
[`loss_circle()`](https://keras3.posit.co/reference/loss_circle.md)  
[`loss_cosine_similarity()`](https://keras3.posit.co/reference/loss_cosine_similarity.md)  
[`loss_ctc()`](https://keras3.posit.co/reference/loss_ctc.md)  
[`loss_dice()`](https://keras3.posit.co/reference/loss_dice.md)  
[`loss_hinge()`](https://keras3.posit.co/reference/loss_hinge.md)  
[`loss_huber()`](https://keras3.posit.co/reference/loss_huber.md)  
[`loss_kl_divergence()`](https://keras3.posit.co/reference/loss_kl_divergence.md)  
[`loss_log_cosh()`](https://keras3.posit.co/reference/loss_log_cosh.md)  
[`loss_mean_absolute_error()`](https://keras3.posit.co/reference/loss_mean_absolute_error.md)  
[`loss_mean_absolute_percentage_error()`](https://keras3.posit.co/reference/loss_mean_absolute_percentage_error.md)  
[`loss_mean_squared_error()`](https://keras3.posit.co/reference/loss_mean_squared_error.md)  
[`loss_mean_squared_logarithmic_error()`](https://keras3.posit.co/reference/loss_mean_squared_logarithmic_error.md)  
[`loss_poisson()`](https://keras3.posit.co/reference/loss_poisson.md)  
[`loss_sparse_categorical_crossentropy()`](https://keras3.posit.co/reference/loss_sparse_categorical_crossentropy.md)  
[`loss_squared_hinge()`](https://keras3.posit.co/reference/loss_squared_hinge.md)  
[`loss_tversky()`](https://keras3.posit.co/reference/loss_tversky.md)  
[`metric_binary_crossentropy()`](https://keras3.posit.co/reference/metric_binary_crossentropy.md)  
[`metric_binary_focal_crossentropy()`](https://keras3.posit.co/reference/metric_binary_focal_crossentropy.md)  
[`metric_categorical_crossentropy()`](https://keras3.posit.co/reference/metric_categorical_crossentropy.md)  
[`metric_categorical_focal_crossentropy()`](https://keras3.posit.co/reference/metric_categorical_focal_crossentropy.md)  
[`metric_categorical_hinge()`](https://keras3.posit.co/reference/metric_categorical_hinge.md)  
[`metric_hinge()`](https://keras3.posit.co/reference/metric_hinge.md)  
[`metric_huber()`](https://keras3.posit.co/reference/metric_huber.md)  
[`metric_kl_divergence()`](https://keras3.posit.co/reference/metric_kl_divergence.md)  
[`metric_mean_absolute_error()`](https://keras3.posit.co/reference/metric_mean_absolute_error.md)  
[`metric_mean_absolute_percentage_error()`](https://keras3.posit.co/reference/metric_mean_absolute_percentage_error.md)  
[`metric_mean_squared_error()`](https://keras3.posit.co/reference/metric_mean_squared_error.md)  
[`metric_mean_squared_logarithmic_error()`](https://keras3.posit.co/reference/metric_mean_squared_logarithmic_error.md)  
[`metric_poisson()`](https://keras3.posit.co/reference/metric_poisson.md)  
[`metric_sparse_categorical_crossentropy()`](https://keras3.posit.co/reference/metric_sparse_categorical_crossentropy.md)  
[`metric_squared_hinge()`](https://keras3.posit.co/reference/metric_squared_hinge.md)  

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
