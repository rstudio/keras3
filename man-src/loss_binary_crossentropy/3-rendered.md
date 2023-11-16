Computes the cross-entropy loss between true labels and predicted labels.

@description
Use this cross-entropy loss for binary (0 or 1) classification applications.
The loss function requires the following inputs:

- `y_true` (true label): This is either 0 or 1.
- `y_pred` (predicted value): This is the model's prediction, i.e, a single
    floating-point value which either represents a
    [logit](https://en.wikipedia.org/wiki/Logit), (i.e, value in `[-inf, inf]`
    when `from_logits=TRUE`) or a probability (i.e, value in `[0., 1.]` when
    `from_logits=FALSE`).

# Examples

```r
y_true <- rbind(c(0, 1), c(0, 0))
y_pred <- rbind(c(0.6, 0.4), c(0.4, 0.6))
loss <- loss_binary_crossentropy(y_true, y_pred)
loss
```

```
## tf.Tensor([0.91629073 0.71355818], shape=(2), dtype=float64)
```
**Recommended Usage:** (set `from_logits=TRUE`)

With `compile()` API:


```r
model %>% compile(
    loss = loss_binary_crossentropy(from_logits=TRUE),
    ...
)
```

As a standalone function:


```r
# Example 1: (batch_size = 1, number of samples = 4)
y_true <- k_array(c(0, 1, 0, 0))
y_pred <- k_array(c(-18.6, 0.51, 2.94, -12.8))
bce <- loss_binary_crossentropy(from_logits = TRUE)
bce(y_true, y_pred)
```

```
## tf.Tensor(0.865458, shape=(), dtype=float32)
```


```r
# Example 2: (batch_size = 2, number of samples = 4)
y_true <- rbind(c(0, 1), c(0, 0))
y_pred <- rbind(c(-18.6, 0.51), c(2.94, -12.8))
# Using default 'auto'/'sum_over_batch_size' reduction type.
bce <- loss_binary_crossentropy(from_logits = TRUE)
bce(y_true, y_pred)
```

```
## tf.Tensor(0.865458, shape=(), dtype=float32)
```

```r
# Using 'sample_weight' attribute
bce(y_true, y_pred, sample_weight = c(0.8, 0.2))
```

```
## tf.Tensor(0.2436386, shape=(), dtype=float32)
```

```r
# 0.243
# Using 'sum' reduction` type.
bce <- loss_binary_crossentropy(from_logits = TRUE, reduction = "sum")
bce(y_true, y_pred)
```

```
## tf.Tensor(1.730916, shape=(), dtype=float32)
```

```r
# Using 'none' reduction type.
bce <- loss_binary_crossentropy(from_logits = TRUE, reduction = NULL)
bce(y_true, y_pred)
```

```
## tf.Tensor([0.23515666 1.4957594 ], shape=(2), dtype=float32)
```

**Default Usage:** (set `from_logits=FALSE`)


```r
# Make the following updates to the above "Recommended Usage" section
# 1. Set `from_logits=FALSE`
loss_binary_crossentropy() # OR ...('from_logits=FALSE')
```

```
## <keras.src.losses.losses.BinaryCrossentropy object>
```

```r
# 2. Update `y_pred` to use probabilities instead of logits
y_pred <- c(0.6, 0.3, 0.2, 0.8) # OR [[0.6, 0.3], [0.2, 0.8]]
```

@returns
Binary crossentropy loss value. shape = `[batch_size, d0, .. dN-1]`.

@param from_logits
Whether to interpret `y_pred` as a tensor of
[logit](https://en.wikipedia.org/wiki/Logit) values. By default, we
assume that `y_pred` is probabilities (i.e., values in `[0, 1)).`

@param label_smoothing
Float in range `[0, 1].` When 0, no smoothing occurs.
When > 0, we compute the loss between the predicted labels
and a smoothed version of the true labels, where the smoothing
squeezes the labels towards 0.5. Larger values of
`label_smoothing` correspond to heavier smoothing.

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
Ground truth values. shape = `[batch_size, d0, .. dN]`.

@param y_pred
The predicted values. shape = `[batch_size, d0, .. dN]`.

@param ...
Passed on to the Python callable

@export
@family losses
@seealso
+ <https:/keras.io/api/losses/probabilistic_losses#binarycrossentropy-class>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/losses/BinaryCrossentropy>

