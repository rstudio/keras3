Update an input by slicing in a tensor of updated values.

@description
At a high level, this operation does
`inputs[start_indices: start_indices + updates.shape] = updates`.
Assume inputs is a tensor of shape `(D0, D1, ..., Dn)`,
`start_indices` must be a list/tuple of n integers, specifying the starting
indices. `updates` must have the same rank as `inputs`, and the size of each
dim must not exceed `Di - start_indices[i]`. For example, if we have 2D
inputs `inputs = np.zeros((5, 5))`, and we want to update the intersection
of last 2 rows and last 2 columns as 1, i.e.,
`inputs[3:, 3:] = np.ones((2, 2))`, then we can use the code below:

```python
inputs = np.zeros((5, 5))
start_indices = [3, 3]
updates = np.ones((2, 2))
inputs = keras.ops.slice_update(inputs, start_indices, updates)
```

@returns
    A tensor, has the same shape and dtype as `inputs`.

@param inputs A tensor, the tensor to be updated.
@param start_indices A list/tuple of shape `(inputs.ndim,)`, specifying
    the starting indices for updating.
@param updates A tensor, the new values to be put to `inputs` at `indices`.
    `updates` must have the same rank as `inputs`.

@export
@family ops
@seealso
+ <https:/keras.io/keras_core/api/ops/core#sliceupdate-function>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/ops/slice_update>
