Computes Kullback-Leibler divergence loss between `y_true` & `y_pred`.

@description
Formula:

```python
loss = y_true * log(y_true / y_pred)
```

Formula:

```python
loss = y_true * log(y_true / y_pred)
```

# Examples
```python
y_true = np.random.randint(0, 2, size=(2, 3)).astype(np.float32)
y_pred = np.random.random(size=(2, 3))
loss = keras.losses.kl_divergence(y_true, y_pred)
assert loss.shape == (2,)
y_true = ops.clip(y_true, 1e-7, 1)
y_pred = ops.clip(y_pred, 1e-7, 1)
assert np.array_equal(
    loss, np.sum(y_true * np.log(y_true / y_pred), axis=-1))
```

@returns
KL Divergence loss values with shape = `[batch_size, d0, .. dN-1]`.

@param reduction Type of reduction to apply to the loss. In almost all cases
    this should be `"sum_over_batch_size"`.
    Supported options are `"sum"`, `"sum_over_batch_size"` or `None`.
@param name Optional name for the loss instance.
@param y_true Tensor of true targets.
@param y_pred Tensor of predicted targets.
@param ... Passed on to the Python callable

@export
@family loss
@seealso
+ <https:/keras.io/api/losses/probabilistic_losses#kldivergence-class>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/losses/KLDivergence>
