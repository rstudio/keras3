# Computes the logarithm of the hyperbolic cosine of the prediction error.

Formula:

    loss <- mean(log(cosh(y_pred - y_true)), axis=-1)

Note that `log(cosh(x))` is approximately equal to `(x ** 2) / 2` for
small `x` and to `abs(x) - log(2)` for large `x`. This means that
'logcosh' works mostly like the mean squared error, but will not be so
strongly affected by the occasional wildly incorrect prediction.

## Usage

``` r
loss_log_cosh(
  y_true,
  y_pred,
  ...,
  reduction = "sum_over_batch_size",
  name = "log_cosh",
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

- reduction:

  Type of reduction to apply to loss. Options are `"sum"`,
  `"sum_over_batch_size"` or `NULL`. Defaults to
  `"sum_over_batch_size"`.

- name:

  Optional name for the instance.

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

Logcosh error values with shape = `[batch_size, d0, .. dN-1]`.

## Examples

    y_true <- rbind(c(0., 1.), c(0., 0.))
    y_pred <- rbind(c(1., 1.), c(0., 0.))
    loss <- loss_log_cosh(y_true, y_pred)
    # 0.108

## See also

- <https://keras.io/api/losses/regression_losses#logcosh-class>

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
