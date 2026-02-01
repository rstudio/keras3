# CTC (Connectionist Temporal Classification) loss.

CTC (Connectionist Temporal Classification) loss.

## Usage

``` r
loss_ctc(
  y_true,
  y_pred,
  ...,
  reduction = "sum_over_batch_size",
  name = "ctc",
  dtype = NULL
)
```

## Arguments

- y_true:

  A tensor of shape `(batch_size, target_max_length)` containing the
  true labels in integer format. `0` always represents the blank/mask
  index and should not be used for classes.

- y_pred:

  A tensor of shape `(batch_size, output_max_length, num_classes)`
  containing logits (the output of your model). They should *not* be
  normalized via softmax.

- ...:

  For forward/backward compatability.

- reduction:

  Type of reduction to apply to the loss. In almost all cases this
  should be `"sum_over_batch_size"`. Supported options are `"sum"`,
  `"sum_over_batch_size"`, `"mean"`, `"mean_with_sample_weight"` or
  `NULL`. `"sum"` sums the loss, `"sum_over_batch_size"` and `"mean"`
  sum the loss and divide by the sample size, and
  `"mean_with_sample_weight"` sums the loss and divides by the sum of
  the sample weights. `"none"` and `NULL` perform no aggregation.
  Defaults to `"sum_over_batch_size"`.

- name:

  Optional name for the loss instance.

- dtype:

  The dtype of the loss's computations. Defaults to `NULL`, which means
  using
  [`config_floatx()`](https://keras3.posit.co/reference/config_floatx.md).
  [`config_floatx()`](https://keras3.posit.co/reference/config_floatx.md)
  is a `"float32"` unless set to different value (via
  [`config_set_floatx()`](https://keras3.posit.co/reference/config_set_floatx.md)).
  If a `keras$DTypePolicy` is provided, then the `compute_dtype` will be
  utilized.

## Value

CTC loss value.

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
[`metric_log_cosh()`](https://keras3.posit.co/reference/metric_log_cosh.md)  
[`metric_mean_absolute_error()`](https://keras3.posit.co/reference/metric_mean_absolute_error.md)  
[`metric_mean_absolute_percentage_error()`](https://keras3.posit.co/reference/metric_mean_absolute_percentage_error.md)  
[`metric_mean_squared_error()`](https://keras3.posit.co/reference/metric_mean_squared_error.md)  
[`metric_mean_squared_logarithmic_error()`](https://keras3.posit.co/reference/metric_mean_squared_logarithmic_error.md)  
[`metric_poisson()`](https://keras3.posit.co/reference/metric_poisson.md)  
[`metric_sparse_categorical_crossentropy()`](https://keras3.posit.co/reference/metric_sparse_categorical_crossentropy.md)  
[`metric_squared_hinge()`](https://keras3.posit.co/reference/metric_squared_hinge.md)  
