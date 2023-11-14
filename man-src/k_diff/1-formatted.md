Calculate the n-th discrete difference along the given axis.

@description
The first difference is given by `out[i] = a[i+1] - a[i]` along
the given axis, higher differences are calculated by using `diff`
recursively.

# Examples
```python
from keras import ops
x = ops.convert_to_tensor([1, 2, 4, 7, 0])
ops.diff(x)
# array([ 1,  2,  3, -7])
ops.diff(x, n=2)
# array([  1,   1, -10])
```

```python
x = ops.convert_to_tensor([[1, 3, 6, 10], [0, 5, 6, 8]])
ops.diff(x)
# array([[2, 3, 4],
#        [5, 1, 2]])
ops.diff(x, axis=0)
# array([[-1,  2,  0, -2]])
```

@returns
Tensor of diagonals.

@param a
Input tensor.

@param n
The number of times values are differenced. Defaults to `1`.

@param axis
Axis to compute discrete difference(s) along.
Defaults to `-1`.(last axis).

@export
@family ops
@seealso
+ <https://www.tensorflow.org/api_docs/python/tf/keras/ops/diff>
