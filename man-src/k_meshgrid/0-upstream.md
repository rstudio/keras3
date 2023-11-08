keras.ops.meshgrid
__signature__
(*x, indexing='xy')
__doc__
Creates grids of coordinates from coordinate vectors.

Given `N` 1-D tensors `T0, T1, ..., TN-1` as inputs with corresponding
lengths `S0, S1, ..., SN-1`, this creates an `N` N-dimensional tensors
`G0, G1, ..., GN-1` each with shape `(S0, ..., SN-1)` where the output
`Gi` is constructed by expanding `Ti` to the result shape.

Args:
    x: 1-D tensors representing the coordinates of a grid.
    indexing: Cartesian (`"xy"`, default) or matrix (`"ij"`) indexing
        of output.

Returns:
    Sequence of N tensors.

Example:
>>> from keras import ops
>>> x = ops.array([1, 2, 3])
>>> y = ops.array([4, 5, 6])

>>> grid_x, grid_y = ops.meshgrid(x, y, indexing="ij")
>>> grid_x
array([[1, 1, 1],
       [2, 2, 2],
       [3, 3, 3]])
>>> grid_y
array([[4, 5, 6],
       [4, 5, 6],
       [4, 5, 6]])
