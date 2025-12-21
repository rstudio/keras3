# Computes the cross-entropy loss between true labels and predicted labels.

Use this cross-entropy loss for binary (0 or 1) classification
applications. The loss function requires the following inputs:

- `y_true` (true label): This is either 0 or 1.

- `y_pred` (predicted value): This is the model's prediction, i.e, a
  single floating-point value which either represents a
  [logit](https://en.wikipedia.org/wiki/Logit), (i.e, value in
  `[-inf, inf]` when `from_logits=TRUE`) or a probability (i.e, value in
  `[0., 1.]` when `from_logits=FALSE`).

## Usage

``` r
loss_binary_crossentropy(
  y_true,
  y_pred,
  from_logits = FALSE,
  label_smoothing = 0,
  axis = -1L,
  ...,
  reduction = "sum_over_batch_size",
  name = "binary_crossentropy",
  dtype = NULL
)
```

## Arguments

- y_true:

  Ground truth values. shape = `[batch_size, d0, .. dN]`.

- y_pred:

  The predicted values. shape = `[batch_size, d0, .. dN]`.

- from_logits:

  Whether to interpret `y_pred` as a tensor of
  [logit](https://en.wikipedia.org/wiki/Logit) values. By default, we
  assume that `y_pred` is probabilities (i.e., values in `[0, 1)).`

- label_smoothing:

  Float in range `[0, 1].` When 0, no smoothing occurs. When \> 0, we
  compute the loss between the predicted labels and a smoothed version
  of the true labels, where the smoothing squeezes the labels towards
  0.5. Larger values of `label_smoothing` correspond to heavier
  smoothing.

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

Binary crossentropy loss value. shape = `[batch_size, d0, .. dN-1]`.

## Examples

    y_true <- rbind(c(0, 1), c(0, 0))
    y_pred <- rbind(c(0.6, 0.4), c(0.4, 0.6))
    loss <- loss_binary_crossentropy(y_true, y_pred)
    loss

    ## tf.Tensor([0.91629076 0.7135582 ], shape=(2), dtype=float32)

**Recommended Usage:** (set `from_logits=TRUE`)

With [`compile()`](https://generics.r-lib.org/reference/compile.html)
API:

    model %>% compile(
        loss = loss_binary_crossentropy(from_logits=TRUE),
        ...
    )

As a standalone function:

    # Example 1: (batch_size = 1, number of samples = 4)
    y_true <- op_array(c(0, 1, 0, 0))
    y_pred <- op_array(c(-18.6, 0.51, 2.94, -12.8))
    bce <- loss_binary_crossentropy(from_logits = TRUE)
    bce(y_true, y_pred)

    ## tf.Tensor(0.865458, shape=(), dtype=float32)

    # Example 2: (batch_size = 2, number of samples = 4)
    y_true <- rbind(c(0, 1), c(0, 0))
    y_pred <- rbind(c(-18.6, 0.51), c(2.94, -12.8))
    # Using default 'auto'/'sum_over_batch_size' reduction type.
    bce <- loss_binary_crossentropy(from_logits = TRUE)
    bce(y_true, y_pred)

    ## tf.Tensor(0.865458, shape=(), dtype=float32)

    # Using 'sample_weight' attribute
    bce(y_true, y_pred, sample_weight = c(0.8, 0.2))

    ## tf.Tensor(0.2436386, shape=(), dtype=float32)

    # 0.243
    # Using 'sum' reduction` type.
    bce <- loss_binary_crossentropy(from_logits = TRUE, reduction = "sum")
    bce(y_true, y_pred)

    ## tf.Tensor(1.730916, shape=(), dtype=float32)

    # Using 'none' reduction type.
    bce <- loss_binary_crossentropy(from_logits = TRUE, reduction = NULL)
    bce(y_true, y_pred)

    ## tf.Tensor([0.23515666 1.4957594 ], shape=(2), dtype=float32)

**Default Usage:** (set `from_logits=FALSE`)

    # Make the following updates to the above "Recommended Usage" section
    # 1. Set `from_logits=FALSE`
    loss_binary_crossentropy() # OR ...('from_logits=FALSE')

    ## <LossFunctionWrapper(<function binary_crossentropy at 0x0>, kwargs={'from_logits': False, 'label_smoothing': 0.0, 'axis': -1})>
    ##  signature: (y_true, y_pred, sample_weight=None)

    # 2. Update `y_pred` to use probabilities instead of logits
    y_pred <- c(0.6, 0.3, 0.2, 0.8) # OR [[0.6, 0.3], [0.2, 0.8]]

## See also

- <https://keras.io/api/losses/probabilistic_losses#binarycrossentropy-class>

Other losses:  
[`Loss()`](https://keras3.posit.co/reference/Loss.md)  
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
[`metric_log_cosh()`](https://keras3.posit.co/reference/metric_log_cosh.md)  
[`metric_mean_absolute_error()`](https://keras3.posit.co/reference/metric_mean_absolute_error.md)  
[`metric_mean_absolute_percentage_error()`](https://keras3.posit.co/reference/metric_mean_absolute_percentage_error.md)  
[`metric_mean_squared_error()`](https://keras3.posit.co/reference/metric_mean_squared_error.md)  
[`metric_mean_squared_logarithmic_error()`](https://keras3.posit.co/reference/metric_mean_squared_logarithmic_error.md)  
[`metric_poisson()`](https://keras3.posit.co/reference/metric_poisson.md)  
[`metric_sparse_categorical_crossentropy()`](https://keras3.posit.co/reference/metric_sparse_categorical_crossentropy.md)  
[`metric_squared_hinge()`](https://keras3.posit.co/reference/metric_squared_hinge.md)  
