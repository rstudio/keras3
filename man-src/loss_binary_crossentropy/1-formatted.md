Computes the cross-entropy loss between true labels and predicted labels.

@description
Use this cross-entropy loss for binary (0 or 1) classification applications.
The loss function requires the following inputs:

- `y_true` (true label): This is either 0 or 1.
- `y_pred` (predicted value): This is the model's prediction, i.e, a single
    floating-point value which either represents a
    `[logit](https://en.wikipedia.org/wiki/Logit), (i.e, value in [-inf, inf]`
    when `from_logits=True`) or a probability (i.e, value in `[0., 1.]` when
    `from_logits=False`).

# Examples
```python
y_true = [[0, 1], [0, 0]]
y_pred = [[0.6, 0.4], [0.4, 0.6]]
loss = keras.losses.binary_crossentropy(y_true, y_pred)
assert loss.shape == (2,)
loss
# array([0.916 , 0.714], dtype=float32)
```
**Recommended Usage:** (set `from_logits=True`)

With `compile()` API:

```python
model.compile(
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    ...
)
```

As a standalone function:

```python
# Example 1: (batch_size = 1, number of samples = 4)
y_true = [0, 1, 0, 0]
y_pred = [-18.6, 0.51, 2.94, -12.8]
bce = keras.losses.BinaryCrossentropy(from_logits=True)
bce(y_true, y_pred)
# 0.865
```

```python
# Example 2: (batch_size = 2, number of samples = 4)
y_true = [[0, 1], [0, 0]]
y_pred = [[-18.6, 0.51], [2.94, -12.8]]
# Using default 'auto'/'sum_over_batch_size' reduction type.
bce = keras.losses.BinaryCrossentropy(from_logits=True)
bce(y_true, y_pred)
# 0.865
# Using 'sample_weight' attribute
bce(y_true, y_pred, sample_weight=[0.8, 0.2])
# 0.243
# Using 'sum' reduction` type.
bce = keras.losses.BinaryCrossentropy(from_logits=True,
    reduction="sum")
bce(y_true, y_pred)
# 1.730
# Using 'none' reduction type.
bce = keras.losses.BinaryCrossentropy(from_logits=True,
    reduction=None)
bce(y_true, y_pred)
# array([0.235, 1.496], dtype=float32)
```

**Default Usage:** (set `from_logits=False`)

```python
# Make the following updates to the above "Recommended Usage" section
# 1. Set `from_logits=False`
keras.losses.BinaryCrossentropy() # OR ...('from_logits=False')
# 2. Update `y_pred` to use probabilities instead of logits
y_pred = [0.6, 0.3, 0.2, 0.8] # OR [[0.6, 0.3], [0.2, 0.8]]
```

@returns
Binary crossentropy loss value. shape = `[batch_size, d0, .. dN-1]`.

@param from_logits
Whether to interpret `y_pred` as a tensor of
[logit](https://en.wikipedia.org/wiki/Logit) values. By default, we
assume that `y_pred` is probabilities (i.e., values in `[0, 1]).`

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
Supported options are `"sum"`, `"sum_over_batch_size"` or `None`.

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
