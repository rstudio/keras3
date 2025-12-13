# Computes the categorical focal crossentropy loss.

Computes the categorical focal crossentropy loss.

## Usage

``` r
metric_categorical_focal_crossentropy(
  y_true,
  y_pred,
  alpha = 0.25,
  gamma = 2,
  from_logits = FALSE,
  label_smoothing = 0,
  axis = -1L
)
```

## Arguments

- y_true:

  Tensor of one-hot true targets.

- y_pred:

  Tensor of predicted targets.

- alpha:

  A weight balancing factor for all classes, default is `0.25` as
  mentioned in the reference. It can be a list of floats or a scalar. In
  the multi-class case, alpha may be set by inverse class frequency by
  using `compute_class_weight` from `sklearn.utils`.

- gamma:

  A focusing parameter, default is `2.0` as mentioned in the reference.
  It helps to gradually reduce the importance given to simple examples
  in a smooth manner. When `gamma` = 0, there is no focal effect on the
  categorical crossentropy.

- from_logits:

  Whether `y_pred` is expected to be a logits tensor. By default, we
  assume that `y_pred` encodes a probability distribution.

- label_smoothing:

  Float in `[0, 1].` If \> `0` then smooth the labels. For example, if
  `0.1`, use `0.1 / num_classes` for non-target labels and
  `0.9 + 0.1 / num_classes` for target labels.

- axis:

  Defaults to `-1`. The dimension along which the entropy is computed.

## Value

Categorical focal crossentropy loss value.

## Examples

    y_true <- rbind(c(0, 1, 0), c(0, 0, 1))
    y_pred <- rbind(c(0.05, 0.9, 0.05), c(0.1, 0.85, 0.05))
    loss <- loss_categorical_focal_crossentropy(y_true, y_pred)
    loss

    ## tf.Tensor([2.6340148e-04 6.7591208e-01], shape=(2), dtype=float32)

## See also

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
[`metric_categorical_hinge()`](https://keras3.posit.co/dev/reference/metric_categorical_hinge.md)  
[`metric_hinge()`](https://keras3.posit.co/dev/reference/metric_hinge.md)  
[`metric_huber()`](https://keras3.posit.co/dev/reference/metric_huber.md)  
[`metric_kl_divergence()`](https://keras3.posit.co/dev/reference/metric_kl_divergence.md)  
[`metric_log_cosh()`](https://keras3.posit.co/dev/reference/metric_log_cosh.md)  
[`metric_mean_absolute_error()`](https://keras3.posit.co/dev/reference/metric_mean_absolute_error.md)  
[`metric_mean_absolute_percentage_error()`](https://keras3.posit.co/dev/reference/metric_mean_absolute_percentage_error.md)  
[`metric_mean_squared_error()`](https://keras3.posit.co/dev/reference/metric_mean_squared_error.md)  
[`metric_mean_squared_logarithmic_error()`](https://keras3.posit.co/dev/reference/metric_mean_squared_logarithmic_error.md)  
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
