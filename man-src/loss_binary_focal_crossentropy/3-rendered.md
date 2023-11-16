Computes focal cross-entropy loss between true labels and predictions.

@description
According to [Lin et al., 2018](https://arxiv.org/pdf/1708.02002.pdf), it
helps to apply a focal factor to down-weight easy examples and focus more on
hard examples. By default, the focal tensor is computed as follows:

`focal_factor = (1 - output)^gamma` for class 1
`focal_factor = output^gamma` for class 0
where `gamma` is a focusing parameter. When `gamma` = 0, there is no focal
effect on the binary crossentropy loss.

If `apply_class_balancing == TRUE`, this function also takes into account a
weight balancing factor for the binary classes 0 and 1 as follows:

`weight = alpha` for class 1 (`target == 1`)
`weight = 1 - alpha` for class 0
where `alpha` is a float in the range of `[0, 1]`.

Binary cross-entropy loss is often used for binary (0 or 1) classification
tasks. The loss function requires the following inputs:

- `y_true` (true label): This is either 0 or 1.
- `y_pred` (predicted value): This is the model's prediction, i.e, a single
    floating-point value which either represents a
    [logit](https://en.wikipedia.org/wiki/Logit), (i.e, value in `[-inf, inf]`
    when `from_logits=TRUE`) or a probability (i.e, value in `[0., 1.]` when
    `from_logits=FALSE`).

According to [Lin et al., 2018](https://arxiv.org/pdf/1708.02002.pdf), it
helps to apply a "focal factor" to down-weight easy examples and focus more
on hard examples. By default, the focal tensor is computed as follows:

`focal_factor = (1 - output) ** gamma` for class 1
`focal_factor = output ** gamma` for class 0
where `gamma` is a focusing parameter. When `gamma=0`, this function is
equivalent to the binary crossentropy loss.

# Examples

```r
y_true <- rbind(c(0, 1), c(0, 0))
y_pred <- rbind(c(0.6, 0.4), c(0.4, 0.6))
loss <- loss_binary_focal_crossentropy(y_true, y_pred, gamma = 2)
loss
```

```
## tf.Tensor([0.32986466 0.20579838], shape=(2), dtype=float64)
```
With the `compile()` API:


```r
model %>% compile(
    loss = loss_binary_focal_crossentropy(
        gamma = 2.0, from_logits = TRUE),
    ...
)
```

As a standalone function:


```r
# Example 1: (batch_size = 1, number of samples = 4)
y_true <- k_array(c(0, 1, 0, 0))
y_pred <- k_array(c(-18.6, 0.51, 2.94, -12.8))
loss <- loss_binary_focal_crossentropy(gamma = 2, from_logits = TRUE)
loss(y_true, y_pred)
```

```
## tf.Tensor(0.6912122, shape=(), dtype=float32)
```


```r
# Apply class weight
loss <- loss_binary_focal_crossentropy(
  apply_class_balancing = TRUE, gamma = 2, from_logits = TRUE)
loss(y_true, y_pred)
```

```
## tf.Tensor(0.5101333, shape=(), dtype=float32)
```


```r
# Example 2: (batch_size = 2, number of samples = 4)
y_true <- rbind(c(0, 1), c(0, 0))
y_pred <- rbind(c(-18.6, 0.51), c(2.94, -12.8))
# Using default 'auto'/'sum_over_batch_size' reduction type.
loss <- loss_binary_focal_crossentropy(
    gamma = 3, from_logits = TRUE)
loss(y_true, y_pred)
```

```
## tf.Tensor(0.6469951, shape=(), dtype=float32)
```


```r
# Apply class weight
loss <- loss_binary_focal_crossentropy(
     apply_class_balancing = TRUE, gamma = 3, from_logits = TRUE)
loss(y_true, y_pred)
```

```
## tf.Tensor(0.48214132, shape=(), dtype=float32)
```


```r
# Using 'sample_weight' attribute with focal effect
loss <- loss_binary_focal_crossentropy(
    gamma = 3, from_logits = TRUE)
loss(y_true, y_pred, sample_weight = c(0.8, 0.2))
```

```
## tf.Tensor(0.13312504, shape=(), dtype=float32)
```


```r
# Apply class weight
loss <- loss_binary_focal_crossentropy(
     apply_class_balancing = TRUE, gamma = 3, from_logits = TRUE)
loss(y_true, y_pred, sample_weight = c(0.8, 0.2))
```

```
## tf.Tensor(0.09735977, shape=(), dtype=float32)
```


```r
# Using 'sum' reduction` type.
loss <- loss_binary_focal_crossentropy(
    gamma = 4, from_logits = TRUE,
    reduction = "sum")
loss(y_true, y_pred)
```

```
## tf.Tensor(1.2218808, shape=(), dtype=float32)
```


```r
# Apply class weight
loss <- loss_binary_focal_crossentropy(
    apply_class_balancing = TRUE, gamma = 4, from_logits = TRUE,
    reduction = "sum")
loss(y_true, y_pred)
```

```
## tf.Tensor(0.9140807, shape=(), dtype=float32)
```


```r
# Using 'none' reduction type.
loss <- loss_binary_focal_crossentropy(
    gamma = 5, from_logits = TRUE,
    reduction = NULL)
loss(y_true, y_pred)
```

```
## tf.Tensor([0.00174837 1.1561027 ], shape=(2), dtype=float32)
```


```r
# Apply class weight
loss <- loss_binary_focal_crossentropy(
    apply_class_balancing = TRUE, gamma = 5, from_logits = TRUE,
    reduction = NULL)
loss(y_true, y_pred)
```

```
## tf.Tensor([4.3709317e-04 8.6707699e-01], shape=(2), dtype=float32)
```

@returns
Binary focal crossentropy loss value
with shape = `[batch_size, d0, .. dN-1]`.

@param apply_class_balancing
A bool, whether to apply weight balancing on the
binary classes 0 and 1.

@param alpha
A weight balancing factor for class 1, default is `0.25` as
mentioned in reference [Lin et al., 2018](
https://arxiv.org/pdf/1708.02002.pdf).  The weight for class 0 is
`1.0 - alpha`.

@param gamma
A focusing parameter used to compute the focal factor, default is
`2.0` as mentioned in the reference
[Lin et al., 2018](https://arxiv.org/pdf/1708.02002.pdf).

@param from_logits
Whether to interpret `y_pred` as a tensor of
[logit](https://en.wikipedia.org/wiki/Logit) values. By default, we
assume that `y_pred` are probabilities (i.e., values in `[0, 1]`).

@param label_smoothing
Float in `[0, 1]`. When `0`, no smoothing occurs.
When > `0`, we compute the loss between the predicted labels
and a smoothed version of the true labels, where the smoothing
squeezes the labels towards `0.5`.
Larger values of `label_smoothing` correspond to heavier smoothing.

@param axis
The axis along which to compute crossentropy (the features axis).
Defaults to `-1`.

@param reduction
Type of reduction to apply to the loss. In almost all cases
this should be `"sum_over_batch_size"`.
Supported options are `"sum"`, `"sum_over_batch_size"` or `NULL`.

@param name
Optional name for the loss instance.

@param y_true
Ground truth values, of shape `(batch_size, d0, .. dN)`.

@param y_pred
The predicted values, of shape `(batch_size, d0, .. dN)`.

@param ...
Passed on to the Python callable

@export
@family losses
@seealso
+ <https://www.tensorflow.org/api_docs/python/tf/keras/losses/BinaryFocalCrossentropy>

