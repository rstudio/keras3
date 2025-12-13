# Computes the alpha balanced focal crossentropy loss.

Use this crossentropy loss function when there are two or more label
classes and if you want to handle class imbalance without using
`class_weights`. We expect labels to be provided in a `one_hot`
representation.

According to [Lin et al., 2018](https://arxiv.org/pdf/1708.02002), it
helps to apply a focal factor to down-weight easy examples and focus
more on hard examples. The general formula for the focal loss (FL) is as
follows:

`FL(p_t) = (1 - p_t)^gamma * log(p_t)`

where `p_t` is defined as follows:
`p_t = output if y_true == 1, else 1 - output`

`(1 - p_t)^gamma` is the `modulating_factor`, where `gamma` is a
focusing parameter. When `gamma` = 0, there is no focal effect on the
cross entropy. `gamma` reduces the importance given to simple examples
in a smooth manner.

The authors use alpha-balanced variant of focal loss (FL) in the paper:
`FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)`

where `alpha` is the weight factor for the classes. If `alpha` = 1, the
loss won't be able to handle class imbalance properly as all classes
will have the same weight. This can be a constant or a list of
constants. If alpha is a list, it must have the same length as the
number of classes.

The formula above can be generalized to:
`FL(p_t) = alpha * (1 - p_t)^gamma * CrossEntropy(y_true, y_pred)`

where minus comes from `CrossEntropy(y_true, y_pred)` (CE).

Extending this to multi-class case is straightforward:
`FL(p_t) = alpha * (1 - p_t) ** gamma * CategoricalCE(y_true, y_pred)`

In the snippet below, there is `num_classes` floating pointing values
per example. The shape of both `y_pred` and `y_true` are
`(batch_size, num_classes)`.

## Usage

``` r
loss_categorical_focal_crossentropy(
  y_true,
  y_pred,
  alpha = 0.25,
  gamma = 2,
  from_logits = FALSE,
  label_smoothing = 0,
  axis = -1L,
  ...,
  reduction = "sum_over_batch_size",
  name = "categorical_focal_crossentropy",
  dtype = NULL
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

  Whether `output` is expected to be a logits tensor. By default, we
  consider that `output` encodes a probability distribution.

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
  [`config_floatx()`](https://keras3.posit.co/dev/reference/config_floatx.md).
  [`config_floatx()`](https://keras3.posit.co/dev/reference/config_floatx.md)
  is a `"float32"` unless set to different value (via
  [`config_set_floatx()`](https://keras3.posit.co/dev/reference/config_set_floatx.md)).
  If a `keras$DTypePolicy` is provided, then the `compute_dtype` will be
  utilized.

## Value

Categorical focal crossentropy loss value.

## Examples

    y_true <- rbind(c(0, 1, 0), c(0, 0, 1))
    y_pred <- rbind(c(0.05, 0.95, 0), c(0.1, 0.8, 0.1))
    loss <- loss_categorical_focal_crossentropy(y_true, y_pred)
    loss

    ## tf.Tensor([3.2058331e-05 4.6627346e-01], shape=(2), dtype=float32)

Standalone usage:

    y_true <- rbind(c(0, 1, 0), c(0, 0, 1))
    y_pred <- rbind(c(0.05, 0.95, 0), c(0.1, 0.8, 0.1))
    # Using 'auto'/'sum_over_batch_size' reduction type.
    cce <- loss_categorical_focal_crossentropy()
    cce(y_true, y_pred)

    ## tf.Tensor(0.23315276, shape=(), dtype=float32)

    # Calling with 'sample_weight'.
    cce(y_true, y_pred, sample_weight = op_array(c(0.3, 0.7)))

    ## tf.Tensor(0.16320053, shape=(), dtype=float32)

    # Using 'sum' reduction type.
    cce <- loss_categorical_focal_crossentropy(reduction = "sum")
    cce(y_true, y_pred)

    ## tf.Tensor(0.46630552, shape=(), dtype=float32)

    # Using 'none' reduction type.
    cce <- loss_categorical_focal_crossentropy(reduction = NULL)
    cce(y_true, y_pred)

    ## tf.Tensor([3.2058331e-05 4.6627346e-01], shape=(2), dtype=float32)

Usage with the
[`compile()`](https://generics.r-lib.org/reference/compile.html) API:

    model %>% compile(
      optimizer = 'adam',
      loss = loss_categorical_focal_crossentropy())

## See also

Other losses:  
[`Loss()`](https://keras3.posit.co/dev/reference/Loss.md)  
[`loss_binary_crossentropy()`](https://keras3.posit.co/dev/reference/loss_binary_crossentropy.md)  
[`loss_binary_focal_crossentropy()`](https://keras3.posit.co/dev/reference/loss_binary_focal_crossentropy.md)  
[`loss_categorical_crossentropy()`](https://keras3.posit.co/dev/reference/loss_categorical_crossentropy.md)  
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
