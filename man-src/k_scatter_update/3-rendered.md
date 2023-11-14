Update inputs via updates at scattered (sparse) indices.

@description
At a high level, this operation does `inputs[indices] = updates`.
Assume `inputs` is a tensor of shape `(D1, D2, ..., Dn)`, there are 2 main
usages of `scatter_update`.

1. `indices` is a 2D tensor of shape `(num_updates, n)`, where `num_updates`
    is the number of updates to perform, and `updates` is a 1D tensor of
    shape `(num_updates)`. For example, if `inputs` is `k_zeros(c(4, 4, 4))`,
    and we want to update `inputs[2, 3, 4]` and `inputs[1, 2, 4]` as 1, then
    we can use:


```r
inputs <- k_zeros(c(4, 4, 4))
indices <- rbind(c(2, 3, 4), c(1, 2, 4))
updates <- k_array(c(1, 1), "float32")
k_scatter_update(inputs, indices, updates)
```

```
## tf.Tensor(
## [[[0. 0. 0. 0.]
##   [0. 0. 0. 1.]
##   [0. 0. 0. 0.]
##   [0. 0. 0. 0.]]
##
##  [[0. 0. 0. 0.]
##   [0. 0. 0. 0.]
##   [0. 0. 0. 1.]
##   [0. 0. 0. 0.]]
##
##  [[0. 0. 0. 0.]
##   [0. 0. 0. 0.]
##   [0. 0. 0. 0.]
##   [0. 0. 0. 0.]]
##
##  [[0. 0. 0. 0.]
##   [0. 0. 0. 0.]
##   [0. 0. 0. 0.]
##   [0. 0. 0. 0.]]], shape=(4, 4, 4), dtype=float32)
```

2 `indices` is a 2D tensor of shape `(num_updates, k)`, where `num_updates`
    is the number of updates to perform, and `k` (`k <= n`) is the size of
    each index in `indices`. `updates` is a `n - k`-D tensor of shape
    `(num_updates, inputs.shape[k:))`. For example, if
    `inputs = k_zeros(c(4, 4, 4))`, and we want to update `inputs[1, 2, ]`
    and `inputs[2, 3, ]` as `[1, 1, 1, 1]`, then `indices` would have shape
    `(num_updates, 2)` (`k = 2`), and `updates` would have shape
    `(num_updates, 4)` (`inputs.shape[2:] = 4`). See the code below:


```r
inputs <- k_zeros(c(4, 4, 4))
indices <- rbind(c(2, 3), c(3, 4))
updates <- k_array(rbind(c(1, 1, 1, 1), c(1, 1, 1, 1)), "float32")
k_scatter_update(inputs, indices, updates)
```

```
## tf.Tensor(
## [[[0. 0. 0. 0.]
##   [0. 0. 0. 0.]
##   [0. 0. 0. 0.]
##   [0. 0. 0. 0.]]
##
##  [[0. 0. 0. 0.]
##   [0. 0. 0. 0.]
##   [1. 1. 1. 1.]
##   [0. 0. 0. 0.]]
##
##  [[0. 0. 0. 0.]
##   [0. 0. 0. 0.]
##   [0. 0. 0. 0.]
##   [1. 1. 1. 1.]]
##
##  [[0. 0. 0. 0.]
##   [0. 0. 0. 0.]
##   [0. 0. 0. 0.]
##   [0. 0. 0. 0.]]], shape=(4, 4, 4), dtype=float32)
```

@returns
    A tensor, has the same shape and dtype as `inputs`.

@param inputs
A tensor, the tensor to be updated.

@param indices
A tensor or list of shape `(N, inputs$ndim)`, specifying
indices to update. `N` is the number of indices to update, must be
equal to the first dimension of `updates`.

@param updates
A tensor, the new values to be put to `inputs` at `indices`.

@export
@family core ops
@family ops
@seealso
+ <https:/keras.io/keras_core/api/ops/core#scatterupdate-function>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/ops/scatter_update>
