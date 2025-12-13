# Computes focal cross-entropy loss between true labels and predictions.

According to [Lin et al., 2018](https://arxiv.org/pdf/1708.02002), it
helps to apply a focal factor to down-weight easy examples and focus
more on hard examples. By default, the focal tensor is computed as
follows:

`focal_factor = (1 - output)^gamma` for class 1
`focal_factor = output^gamma` for class 0 where `gamma` is a focusing
parameter. When `gamma` = 0, there is no focal effect on the binary
crossentropy loss.

If `apply_class_balancing == TRUE`, this function also takes into
account a weight balancing factor for the binary classes 0 and 1 as
follows:

`weight = alpha` for class 1 (`target == 1`) `weight = 1 - alpha` for
class 0 where `alpha` is a float in the range of `[0, 1]`.

Binary cross-entropy loss is often used for binary (0 or 1)
classification tasks. The loss function requires the following inputs:

- `y_true` (true label): This is either 0 or 1.

- `y_pred` (predicted value): This is the model's prediction, i.e, a
  single floating-point value which either represents a
  [logit](https://en.wikipedia.org/wiki/Logit), (i.e, value in
  `[-inf, inf]` when `from_logits=TRUE`) or a probability (i.e, value in
  `[0., 1.]` when `from_logits=FALSE`).

According to [Lin et al., 2018](https://arxiv.org/pdf/1708.02002), it
helps to apply a "focal factor" to down-weight easy examples and focus
more on hard examples. By default, the focal tensor is computed as
follows:

`focal_factor = (1 - output) ** gamma` for class 1
`focal_factor = output ** gamma` for class 0 where `gamma` is a focusing
parameter. When `gamma=0`, this function is equivalent to the binary
crossentropy loss.

## Usage

``` r
loss_binary_focal_crossentropy(
  y_true,
  y_pred,
  apply_class_balancing = FALSE,
  alpha = 0.25,
  gamma = 2,
  from_logits = FALSE,
  label_smoothing = 0,
  axis = -1L,
  ...,
  reduction = "sum_over_batch_size",
  name = "binary_focal_crossentropy",
  dtype = NULL
)
```

## Arguments

- y_true:

  Ground truth values, of shape `(batch_size, d0, .. dN)`.

- y_pred:

  The predicted values, of shape `(batch_size, d0, .. dN)`.

- apply_class_balancing:

  A bool, whether to apply weight balancing on the binary classes 0 and
  1.

- alpha:

  A weight balancing factor for class 1, default is `0.25` as mentioned
  in reference [Lin et al., 2018](https://arxiv.org/pdf/1708.02002). The
  weight for class 0 is `1.0 - alpha`.

- gamma:

  A focusing parameter used to compute the focal factor, default is
  `2.0` as mentioned in the reference [Lin et al.,
  2018](https://arxiv.org/pdf/1708.02002).

- from_logits:

  Whether to interpret `y_pred` as a tensor of
  [logit](https://en.wikipedia.org/wiki/Logit) values. By default, we
  assume that `y_pred` are probabilities (i.e., values in `[0, 1]`).

- label_smoothing:

  Float in `[0, 1]`. When `0`, no smoothing occurs. When \> `0`, we
  compute the loss between the predicted labels and a smoothed version
  of the true labels, where the smoothing squeezes the labels towards
  `0.5`. Larger values of `label_smoothing` correspond to heavier
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
  [`config_floatx()`](https://keras3.posit.co/dev/reference/config_floatx.md).
  [`config_floatx()`](https://keras3.posit.co/dev/reference/config_floatx.md)
  is a `"float32"` unless set to different value (via
  [`config_set_floatx()`](https://keras3.posit.co/dev/reference/config_set_floatx.md)).
  If a `keras$DTypePolicy` is provided, then the `compute_dtype` will be
  utilized.

## Value

Binary focal crossentropy loss value with shape =
`[batch_size, d0, .. dN-1]`.

## Examples

    y_true <- rbind(c(0, 1), c(0, 0))
    y_pred <- rbind(c(0.6, 0.4), c(0.4, 0.6))
    loss <- loss_binary_focal_crossentropy(y_true, y_pred, gamma = 2)
    loss

    ## tf.Tensor([0.32986468 0.20579839], shape=(2), dtype=float32)

With the
[`compile()`](https://generics.r-lib.org/reference/compile.html) API:

    model %>% compile(
        loss = loss_binary_focal_crossentropy(
            gamma = 2.0, from_logits = TRUE),
        ...
    )

As a standalone function:

    # Example 1: (batch_size = 1, number of samples = 4)
    y_true <- op_array(c(0, 1, 0, 0))
    y_pred <- op_array(c(-18.6, 0.51, 2.94, -12.8))
    loss <- loss_binary_focal_crossentropy(gamma = 2, from_logits = TRUE)
    loss(y_true, y_pred)

    ## tf.Tensor(0.6912122, shape=(), dtype=float32)

    # Apply class weight
    loss <- loss_binary_focal_crossentropy(
      apply_class_balancing = TRUE, gamma = 2, from_logits = TRUE)
    loss(y_true, y_pred)

    ## tf.Tensor(0.5101333, shape=(), dtype=float32)

    # Example 2: (batch_size = 2, number of samples = 4)
    y_true <- rbind(c(0, 1), c(0, 0))
    y_pred <- rbind(c(-18.6, 0.51), c(2.94, -12.8))
    # Using default 'auto'/'sum_over_batch_size' reduction type.
    loss <- loss_binary_focal_crossentropy(
        gamma = 3, from_logits = TRUE)
    loss(y_true, y_pred)

    ## tf.Tensor(0.6469951, shape=(), dtype=float32)

    # Apply class weight
    loss <- loss_binary_focal_crossentropy(
         apply_class_balancing = TRUE, gamma = 3, from_logits = TRUE)
    loss(y_true, y_pred)

    ## tf.Tensor(0.48214132, shape=(), dtype=float32)

    # Using 'sample_weight' attribute with focal effect
    loss <- loss_binary_focal_crossentropy(
        gamma = 3, from_logits = TRUE)
    loss(y_true, y_pred, sample_weight = c(0.8, 0.2))

    ## tf.Tensor(0.13312504, shape=(), dtype=float32)

    # Apply class weight
    loss <- loss_binary_focal_crossentropy(
         apply_class_balancing = TRUE, gamma = 3, from_logits = TRUE)
    loss(y_true, y_pred, sample_weight = c(0.8, 0.2))

    ## tf.Tensor(0.09735977, shape=(), dtype=float32)

    # Using 'sum' reduction` type.
    loss <- loss_binary_focal_crossentropy(
        gamma = 4, from_logits = TRUE,
        reduction = "sum")
    loss(y_true, y_pred)

    ## tf.Tensor(1.2218808, shape=(), dtype=float32)

    # Apply class weight
    loss <- loss_binary_focal_crossentropy(
        apply_class_balancing = TRUE, gamma = 4, from_logits = TRUE,
        reduction = "sum")
    loss(y_true, y_pred)

    ## tf.Tensor(0.9140807, shape=(), dtype=float32)

    # Using 'none' reduction type.
    loss <- loss_binary_focal_crossentropy(
        gamma = 5, from_logits = TRUE,
        reduction = NULL)
    loss(y_true, y_pred)

    ## tf.Tensor([0.00174837 1.1561027 ], shape=(2), dtype=float32)

    # Apply class weight
    loss <- loss_binary_focal_crossentropy(
        apply_class_balancing = TRUE, gamma = 5, from_logits = TRUE,
        reduction = NULL)
    loss(y_true, y_pred)

    ## tf.Tensor([4.3709317e-04 8.6707699e-01], shape=(2), dtype=float32)

## See also

Other losses:  
[`Loss()`](https://keras3.posit.co/dev/reference/Loss.md)  
[`loss_binary_crossentropy()`](https://keras3.posit.co/dev/reference/loss_binary_crossentropy.md)  
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
