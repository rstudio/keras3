# Computes the crossentropy loss between the labels and predictions.

Use this crossentropy loss function when there are two or more label
classes. We expect labels to be provided in a `one_hot` representation.
If you want to provide labels as integers, please use
`SparseCategoricalCrossentropy` loss. There should be `num_classes`
floating point values per feature, i.e., the shape of both `y_pred` and
`y_true` are `[batch_size, num_classes]`.

## Usage

``` r
loss_categorical_crossentropy(
  y_true,
  y_pred,
  from_logits = FALSE,
  label_smoothing = 0,
  axis = -1L,
  ...,
  reduction = "sum_over_batch_size",
  name = "categorical_crossentropy",
  dtype = NULL
)
```

## Arguments

- y_true:

  Tensor of one-hot true targets.

- y_pred:

  Tensor of predicted targets.

- from_logits:

  Whether `y_pred` is expected to be a logits tensor. By default, we
  assume that `y_pred` encodes a probability distribution.

- label_smoothing:

  Float in `[0, 1].` When \> 0, label values are smoothed, meaning the
  confidence on label values are relaxed. For example, if `0.1`, use
  `0.1 / num_classes` for non-target labels and
  `0.9 + 0.1 / num_classes` for target labels.

- axis:

  The axis along which to compute crossentropy (the features axis).
  Defaults to `-1`.

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

Categorical crossentropy loss value.

## Examples

    y_true <- rbind(c(0, 1, 0), c(0, 0, 1))
    y_pred <- rbind(c(0.05, 0.95, 0), c(0.1, 0.8, 0.1))
    loss <- loss_categorical_crossentropy(y_true, y_pred)
    loss

    ## tf.Tensor([0.05129331 2.3025851 ], shape=(2), dtype=float32)

Standalone usage:

    y_true <- rbind(c(0, 1, 0), c(0, 0, 1))
    y_pred <- rbind(c(0.05, 0.95, 0), c(0.1, 0.8, 0.1))
    # Using 'auto'/'sum_over_batch_size' reduction type.
    cce <- loss_categorical_crossentropy()
    cce(y_true, y_pred)

    ## tf.Tensor(1.1769392, shape=(), dtype=float32)

    # Calling with 'sample_weight'.
    cce(y_true, y_pred, sample_weight = op_array(c(0.3, 0.7)))

    ## tf.Tensor(0.8135988, shape=(), dtype=float32)

    # Using 'sum' reduction type.
    cce <- loss_categorical_crossentropy(reduction = "sum")
    cce(y_true, y_pred)

    ## tf.Tensor(2.3538785, shape=(), dtype=float32)

    # Using 'none' reduction type.
    cce <- loss_categorical_crossentropy(reduction = NULL)
    cce(y_true, y_pred)

    ## tf.Tensor([0.05129331 2.3025851 ], shape=(2), dtype=float32)

Usage with the
[`compile()`](https://generics.r-lib.org/reference/compile.html) API:

    model %>% compile(optimizer = 'sgd',
                  loss=loss_categorical_crossentropy())

## See also

- <https://keras.io/api/losses/probabilistic_losses#categoricalcrossentropy-class>

Other losses:  
[`Loss()`](https://keras3.posit.co/reference/Loss.md)  
[`loss_binary_crossentropy()`](https://keras3.posit.co/reference/loss_binary_crossentropy.md)  
[`loss_binary_focal_crossentropy()`](https://keras3.posit.co/reference/loss_binary_focal_crossentropy.md)  
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
[`metric_log_cosh()`](https://keras3.posit.co/reference/metric_log_cosh.md)  
[`metric_mean_absolute_error()`](https://keras3.posit.co/reference/metric_mean_absolute_error.md)  
[`metric_mean_absolute_percentage_error()`](https://keras3.posit.co/reference/metric_mean_absolute_percentage_error.md)  
[`metric_mean_squared_error()`](https://keras3.posit.co/reference/metric_mean_squared_error.md)  
[`metric_mean_squared_logarithmic_error()`](https://keras3.posit.co/reference/metric_mean_squared_logarithmic_error.md)  
[`metric_poisson()`](https://keras3.posit.co/reference/metric_poisson.md)  
[`metric_sparse_categorical_crossentropy()`](https://keras3.posit.co/reference/metric_sparse_categorical_crossentropy.md)  
[`metric_squared_hinge()`](https://keras3.posit.co/reference/metric_squared_hinge.md)  
