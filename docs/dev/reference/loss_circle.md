# Computes Circle Loss between integer labels and L2-normalized embeddings.

It is designed to minimize within-class distances and maximize
between-class distances in L2 normalized embedding space.

This is a metric learning loss designed to minimize within-class
distance and maximize between-class distance in a flexible manner by
dynamically adjusting the penalty strength based on optimization status
of each similarity score.

To use Circle Loss effectively, the model should output embeddings
without an activation function (such as a `Dense` layer with
`activation=NULL`) followed by `UnitNormalization` layer to ensure
unit-norm embeddings.

## Usage

``` r
loss_circle(
  y_true,
  y_pred,
  ref_labels = NULL,
  ref_embeddings = NULL,
  remove_diagonal = TRUE,
  gamma = 80L,
  margin = 0.4,
  ...,
  reduction = "sum_over_batch_size",
  name = "circle",
  dtype = NULL
)
```

## Arguments

- y_true:

  Tensor with ground truth labels in integer format.

- y_pred:

  Tensor with predicted L2 normalized embeddings.

- ref_labels:

  Optional integer tensor with labels for reference embeddings. If
  `NULL`, defaults to `y_true`.

- ref_embeddings:

  Optional tensor with L2 normalized reference embeddings. If `NULL`,
  defaults to `y_pred`.

- remove_diagonal:

  Boolean, whether to remove self-similarities from the positive mask.
  Defaults to `TRUE`.

- gamma:

  Scaling factor that determines the largest scale of each similarity
  score. Defaults to `80`.

- margin:

  The relaxation factor, below this distance, negatives are up weighted
  and positives are down weighted. Similarly, above this distance
  negatives are down weighted and positive are up weighted. Defaults to
  `0.4`.

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
  [`config_floatx()`](https://keras3.posit.co/dev/reference/config_floatx.md).
  [`config_floatx()`](https://keras3.posit.co/dev/reference/config_floatx.md)
  is a `"float32"` unless set to different value (via
  [`config_set_floatx()`](https://keras3.posit.co/dev/reference/config_set_floatx.md)).
  If a `keras.DTypePolicy` is provided, then the `compute_dtype` will be
  utilized.

## Value

Circle loss value.

## Examples

Usage with the
[`compile()`](https://generics.r-lib.org/reference/compile.html) API:

    model <- keras_model_sequential(input_shape = c(224, 224, 3)) |>
      layer_conv_2d(16, c(3, 3), activation = 'relu') |>
      layer_flatten() |>
      layer_dense(64, activation = NULL) |>   # No activation
      layer_unit_normalization()  # L2 normalization

    model |>
      compile(optimizer = "adam", loss = loss_circle())

## Reference

- [Yifan Sun et al., 2020](https://arxiv.org/abs/2002.10857)

## See also

Other losses:  
[`Loss()`](https://keras3.posit.co/dev/reference/Loss.md)  
[`loss_binary_crossentropy()`](https://keras3.posit.co/dev/reference/loss_binary_crossentropy.md)  
[`loss_binary_focal_crossentropy()`](https://keras3.posit.co/dev/reference/loss_binary_focal_crossentropy.md)  
[`loss_categorical_crossentropy()`](https://keras3.posit.co/dev/reference/loss_categorical_crossentropy.md)  
[`loss_categorical_focal_crossentropy()`](https://keras3.posit.co/dev/reference/loss_categorical_focal_crossentropy.md)  
[`loss_categorical_generalized_cross_entropy()`](https://keras3.posit.co/dev/reference/loss_categorical_generalized_cross_entropy.md)  
[`loss_categorical_hinge()`](https://keras3.posit.co/dev/reference/loss_categorical_hinge.md)  
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
[`metric_mean_squared_logarithmic_error()`](https://keras3.posit.co/dev/reference/metric_mean_squared_logarithmic_error.md)  
[`metric_poisson()`](https://keras3.posit.co/dev/reference/metric_poisson.md)  
[`metric_sparse_categorical_crossentropy()`](https://keras3.posit.co/dev/reference/metric_sparse_categorical_crossentropy.md)  
[`metric_squared_hinge()`](https://keras3.posit.co/dev/reference/metric_squared_hinge.md)  
