# Computes the crossentropy loss between the labels and predictions.

Use this crossentropy loss function when there are two or more label
classes. We expect labels to be provided as integers. If you want to
provide labels using `one-hot` representation, please use
`CategoricalCrossentropy` loss. There should be `# classes` floating
point values per feature for `y_pred` and a single floating point value
per feature for `y_true`.

In the snippet below, there is a single floating point value per example
for `y_true` and `num_classes` floating pointing values per example for
`y_pred`. The shape of `y_true` is `[batch_size]` and the shape of
`y_pred` is `[batch_size, num_classes]`.

## Usage

``` r
loss_sparse_categorical_crossentropy(
  y_true,
  y_pred,
  from_logits = FALSE,
  ignore_class = NULL,
  axis = -1L,
  ...,
  reduction = "sum_over_batch_size",
  name = "sparse_categorical_crossentropy",
  dtype = NULL
)
```

## Arguments

- y_true:

  Ground truth values.

- y_pred:

  The predicted values.

- from_logits:

  Whether `y_pred` is expected to be a logits tensor. By default, we
  assume that `y_pred` encodes a probability distribution.

- ignore_class:

  Optional integer. The ID of a class to be ignored during loss
  computation. This is useful, for example, in segmentation problems
  featuring a "void" class (commonly -1 or 255) in segmentation maps. By
  default (`ignore_class=NULL`), all classes are considered.

- axis:

  Defaults to `-1`. The dimension along which the entropy is computed.

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
  If a `keras$DTypePolicy` is provided, then the `compute_dtype` will be
  utilized.

## Value

Sparse categorical crossentropy loss value.

## Examples

    y_true <- op_array(c(1L, 2L))
    y_pred <- op_array(rbind(c(0.05, 0.95, 0), c(0.1, 0.8, 0.1)))
    loss <- loss_sparse_categorical_crossentropy(y_true, y_pred)
    loss

    ## tf.Tensor([0.05129339 2.30258509], shape=(2), dtype=float64)

    y_true <- op_array(c(1L, 2L))
    y_pred <- op_array(rbind(c(0.05, 0.95, 0), c(0.1, 0.8, 0.1)))
    # Using 'auto'/'sum_over_batch_size' reduction type.
    scce <- loss_sparse_categorical_crossentropy()
    scce(y_true, y_pred)

    ## tf.Tensor(1.1769392, shape=(), dtype=float32)

    # Calling with 'sample_weight'.
    scce(y_true, y_pred, sample_weight = op_array(c(0.3, 0.7)))

    ## tf.Tensor(0.8135988, shape=(), dtype=float32)

    # Using 'sum' reduction type.
    scce <- loss_sparse_categorical_crossentropy(reduction="sum")
    scce(y_true, y_pred)

    ## tf.Tensor(2.3538785, shape=(), dtype=float32)

    # Using 'none' reduction type.
    scce <- loss_sparse_categorical_crossentropy(reduction=NULL)
    scce(y_true, y_pred)

    ## tf.Tensor([0.05129344 2.3025851 ], shape=(2), dtype=float32)

Usage with the
[`compile()`](https://generics.r-lib.org/reference/compile.html) API:

    model %>% compile(optimizer = 'sgd',
                      loss = loss_sparse_categorical_crossentropy())

## See also

- <https://keras.io/api/losses/probabilistic_losses#sparsecategoricalcrossentropy-class>

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
