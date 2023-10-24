Computes Kullback-Leibler divergence loss between `y_true` & `y_pred`.

Formula:

```python
loss = y_true * log(y_true / y_pred)
```

Args:
    y_true: Tensor of true targets.
    y_pred: Tensor of predicted targets.

Returns:
    KL Divergence loss values with shape = `[batch_size, d0, .. dN-1]`.

Example:

>>> y_true = np.random.randint(0, 2, size=(2, 3)).astype(np.float32)
>>> y_pred = np.random.random(size=(2, 3))
>>> loss = keras.losses.kl_divergence(y_true, y_pred)
>>> assert loss.shape == (2,)
>>> y_true = ops.clip(y_true, 1e-7, 1)
>>> y_pred = ops.clip(y_pred, 1e-7, 1)
>>> assert np.array_equal(
...     loss, np.sum(y_true * np.log(y_true / y_pred), axis=-1))
