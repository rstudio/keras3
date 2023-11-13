Creates grids of coordinates from coordinate vectors.

@description
Given `N` 1-D tensors `T0, T1, ..., TN-1` as inputs with corresponding
lengths `S0, S1, ..., SN-1`, this creates an `N` N-dimensional tensors
`G0, G1, ..., GN-1` each with shape `(S0, ..., SN-1)` where the output
`Gi` is constructed by expanding `Ti` to the result shape.

# Examples

```r
x <- k_array(c(1, 2, 3), "int32")
y <- k_array(c(4, 5, 6), "int32")
```


```r
c(grid_x, grid_y) %<-% k_meshgrid(x, y, indexing = "ij")
grid_x
```

```
## tf.Tensor(
## [[1 1 1]
##  [2 2 2]
##  [3 3 3]], shape=(3, 3), dtype=int32)
```

```r
# array([[1, 1, 1],
#        [2, 2, 2],
#        [3, 3, 3]))
grid_y
```

```
## tf.Tensor(
## [[4 5 6]
##  [4 5 6]
##  [4 5 6]], shape=(3, 3), dtype=int32)
```

```r
# array([[4, 5, 6],
#        [4, 5, 6],
#        [4, 5, 6]))
```

@returns
Sequence of N tensors.

@param ... 1-D tensors representing the coordinates of a grid.
@param indexing Cartesian (`"xy"`, default) or matrix (`"ij"`) indexing
    of output.

@export
@family ops
@seealso
+ <https:/keras.io/keras_core/api/ops/numpy#meshgrid-function>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/ops/meshgrid>

