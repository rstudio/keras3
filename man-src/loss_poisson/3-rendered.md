Computes the Poisson loss between `y_true` & `y_pred`.

@description
Formula:

```python
loss = y_pred - y_true * log(y_pred)
```
Formula:

```python
loss = y_pred - y_true * log(y_pred)
```

# Examples
```python
y_true = np.random.randint(0, 2, size=(2, 3))
y_pred = np.random.random(size=(2, 3))
loss = keras.losses.poisson(y_true, y_pred)
assert loss.shape == (2,)
y_pred = y_pred + 1e-7
assert np.allclose(
    loss, np.mean(y_pred - y_true * np.log(y_pred), axis=-1),
    atol=1e-5)
```

@returns
Poisson loss values with shape = `[batch_size, d0, .. dN-1]`.

@param reduction Type of reduction to apply to the loss. In almost all cases
    this should be `"sum_over_batch_size"`.
    Supported options are `"sum"`, `"sum_over_batch_size"` or `None`.
@param name Optional name for the loss instance.
@param y_true Ground truth values. shape = `[batch_size, d0, .. dN]`.
@param y_pred The predicted values. shape = `[batch_size, d0, .. dN]`.
@param ... Passed on to the Python callable

@export
@family loss
@seealso
+ <https:/keras.io/api/losses/probabilistic_losses#poisson-class>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/losses/Poisson>
