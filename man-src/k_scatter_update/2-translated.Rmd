Update inputs via updates at scattered (sparse) indices.

@description
At a high level, this operation does `inputs[indices] = updates`.
Assume `inputs` is a tensor of shape `(D0, D1, ..., Dn)`, there are 2 main
usages of `scatter_update`.

1. `indices` is a 2D tensor of shape `(num_updates, n)`, where `num_updates`
    is the number of updates to perform, and `updates` is a 1D tensor of
    shape `(num_updates,)`. For example, if `inputs` is `zeros((4, 4, 4))`,
    and we want to update `inputs[1, 2, 3]` and `inputs[0, 1, 3]` as 1, then
    we can use:

```python
inputs = np.zeros((4, 4, 4))
indices = [[1, 2, 3], [0, 1, 3]]
updates = np.array([1., 1.])
inputs = keras.ops.scatter_update(inputs, indices, updates)
```

2 `indices` is a 2D tensor of shape `(num_updates, k)`, where `num_updates`
    is the number of updates to perform, and `k` (`k < n`) is the size of
    each index in `indices`. `updates` is a `n - k`-D tensor of shape
    `(num_updates, inputs.shape[k:])`. For example, if
    `inputs = np.zeros((4, 4, 4))`, and we want to update `inputs[1, 2, :]`
    and `inputs[2, 3, :]` as `[1, 1, 1, 1]`, then `indices` would have shape
    `(num_updates, 2)` (`k = 2`), and `updates` would have shape
    `(num_updates, 4)` (`inputs.shape[2:] = 4`). See the code below:

```python
inputs = np.zeros((4, 4, 4))
indices = [[1, 2], [2, 3]]
updates = np.array([[1., 1., 1, 1,], [1., 1., 1, 1,])
inputs = keras.ops.scatter_update(inputs, indices, updates)
```

@returns
    A tensor, has the same shape and dtype as `inputs`.

@param inputs A tensor, the tensor to be updated.
@param indices A tensor or list/tuple of shape `(N, inputs.ndim)`, specifying
    indices to update. `N` is the number of indices to update, must be
    equal to the first dimension of `updates`.
@param updates A tensor, the new values to be put to `inputs` at `indices`.

@export
@family ops
@seealso
+ <https://www.tensorflow.org/api_docs/python/tf/keras/ops/scatter_update>
