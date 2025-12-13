# Computes the Dice loss value between `y_true` and `y_pred`.

Formula:

    loss = 1 - (2 * sum(y_true * y_pred)) / (sum(y_true) + sum(y_pred))

Formula:

    loss = 1 - (2 * sum(y_true * y_pred)) / (sum(y_true) + sum(y_pred))

## Usage

``` r
loss_dice(
  y_true,
  y_pred,
  ...,
  reduction = "sum_over_batch_size",
  name = "dice",
  axis = NULL,
  dtype = NULL
)
```

## Arguments

- y_true:

  Tensor of true targets.

- y_pred:

  Tensor of predicted targets.

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

  String, name for the object

- axis:

  List of which dimensions the loss is calculated. Defaults to `NULL`.

- dtype:

  The dtype of the loss's computations. Defaults to `NULL`, which means
  using
  [`config_floatx()`](https://keras3.posit.co/dev/reference/config_floatx.md).
  [`config_floatx()`](https://keras3.posit.co/dev/reference/config_floatx.md)
  is a `"float32"` unless set to different value (via
  [`config_set_floatx()`](https://keras3.posit.co/dev/reference/config_set_floatx.md)).
  If a `keras$DTypePolicy` is provided, then the `compute_dtype` will be
  utilized.

## Value

if `y_true` and `y_pred` are provided, Dice loss value. Otherwise, a
[`Loss()`](https://keras3.posit.co/dev/reference/Loss.md) instance.

## Example

    y_true <- array(c(1, 1, 0, 0,
                      1, 1, 0, 0), dim = c(2, 2, 2, 1))
    y_pred <- array(c(0, 0.4, 0,   0,
                      1,   0, 1, 0.9), dim = c(2, 2, 2, 1))

    axis <- c(2, 3, 4)
    loss_fn <- loss_dice(axis = axis, reduction = NULL)
    loss <- loss_fn(y_true, y_pred)
    stopifnot(shape(loss) == shape(2))
    loss

    ## tf.Tensor([0.5        0.75757575], shape=(2), dtype=float32)

    loss_fn <- loss_dice()
    loss <- loss_fn(y_true, y_pred)
    stopifnot(shape(loss) == shape())
    loss

    ## tf.Tensor(0.6164384, shape=(), dtype=float32)

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
