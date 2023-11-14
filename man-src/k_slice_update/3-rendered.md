Update an input by slicing in a tensor of updated values.

@description
At a high level, this operation does
`inputs[start_indices: start_indices + updates.shape] = updates`.
Assume inputs is a tensor of shape `(D1, D2, ..., Dn)`,
`start_indices` must be a list of n integers, specifying the starting
indices. `updates` must have the same rank as `inputs`, and the size of each
dim must not exceed `Di - start_indices[i]`. For example, if we have 2D
inputs `inputs = k_zeros(c(5, 5))`, and we want to update the intersection
of last 2 rows and last 2 columns as 1, i.e.,
`inputs[4:5, 4:5] = k_ones(c(2, 2))`, then we can use the code below:


```r
inputs <- k_zeros(c(5, 5))
start_indices <- c(3, 3)
updates <- k_ones(c(2, 2))
k_slice_update(inputs, start_indices, updates)
```

```
## tf.Tensor(
## [[0. 0. 0. 0. 0.]
##  [0. 0. 0. 0. 0.]
##  [0. 0. 1. 1. 0.]
##  [0. 0. 1. 1. 0.]
##  [0. 0. 0. 0. 0.]], shape=(5, 5), dtype=float32)
```

@returns
    A tensor, has the same shape and dtype as `inputs`.

@param inputs
A tensor, the tensor to be updated.

@param start_indices
A list of length `inputs$ndim`, specifying
the starting indices for updating.

@param updates
A tensor, the new values to be put to `inputs` at `indices`.
`updates` must have the same rank as `inputs`.

@export
@family ops
@seealso
+ <https:/keras.io/keras_core/api/ops/core#sliceupdate-function>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/ops/slice_update>
