Return specified diagonals.

@description
If `x` is 2-D, returns the diagonal of `x` with the given offset, i.e., the
collection of elements of the form `x[i, i+offset]`.

If `x` has more than two dimensions, the axes specified by `axis1`
and `axis2` are used to determine the 2-D sub-array whose diagonal
is returned.

The shape of the resulting array can be determined by removing `axis1`
and `axis2` and appending an index to the right equal to the size of
the resulting diagonals.

# Examples
```python
from keras import ops
x = ops.arange(4).reshape((2, 2))
x
# array([[0, 1],
#        [2, 3]])
x.diagonal()
# array([0, 3])
x.diagonal(1)
# array([1])
```

```python
x = ops.arange(8).reshape((2, 2, 2))
x
# array([[[0, 1],
#         [2, 3]],
#        [[4, 5],
#         [6, 7]]])
x.diagonal(0, 0, 1)
# array([[0, 6],
#        [1, 7]])
```

@returns
Tensor of diagonals.

@param x
Input tensor.

@param offset
Offset of the diagonal from the main diagonal.
Can be positive or negative. Defaults to `0`.(main diagonal).

@param axis1
Axis to be used as the first axis of the 2-D sub-arrays.
Defaults to `0`.(first axis).

@param axis2
Axis to be used as the second axis of the 2-D sub-arrays.
Defaults to `1` (second axis).

@export
@family ops
@seealso
+ <https:/keras.io/keras_core/api/ops/numpy#diagonal-function>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/ops/diagonal>
