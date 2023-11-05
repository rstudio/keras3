Computes the squared hinge loss between `y_true` & `y_pred`.

@description
Formula:

```python
loss = mean(square(maximum(1 - y_true * y_pred, 0)), axis=-1)
```
Formula:

```python
loss = square(maximum(1 - y_true * y_pred, 0))
```

`y_true` values are expected to be -1 or 1. If binary (0 or 1) labels are
provided we will convert them to -1 or 1.

# Returns
Squared hinge loss values with shape = `[batch_size, d0, .. dN-1]`.

# Examples
```python
y_true = np.random.choice([-1, 1], size=(2, 3))
y_pred = np.random.random(size=(2, 3))
loss = keras.losses.squared_hinge(y_true, y_pred)
```

@param reduction Type of reduction to apply to the loss. In almost all cases
    this should be `"sum_over_batch_size"`.
    Supported options are `"sum"`, `"sum_over_batch_size"` or `None`.
@param name Optional name for the loss instance.
@param y_true The ground truth values. `y_true` values are expected to be -1
    or 1. If binary (0 or 1) labels are provided we will convert them
    to -1 or 1 with shape = `[batch_size, d0, .. dN]`.
@param y_pred The predicted values with shape = `[batch_size, d0, .. dN]`.
@param ... Passed on to the Python callable

@export
@family loss
@seealso
+ <https://www.tensorflow.org/api_docs/python/tf/keras/losses/SquaredHinge>
