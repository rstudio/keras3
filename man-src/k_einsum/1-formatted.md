Evaluates the Einstein summation convention on the operands.

# Returns
The calculation based on the Einstein summation convention.

# Examples
```python
from keras import ops
a = ops.arange(25).reshape(5, 5)
b = ops.arange(5)
c = ops.arange(6).reshape(2, 3)
```

Trace of a matrix:

```python
ops.einsum("ii", a)
# 60
ops.einsum(a, [0, 0])
# 60
ops.trace(a)
# 60
```

Extract the diagonal:

```python
ops.einsum("ii -> i", a)
# array([ 0,  6, 12, 18, 24])
ops.einsum(a, [0, 0], [0])
# array([ 0,  6, 12, 18, 24])
ops.diag(a)
# array([ 0,  6, 12, 18, 24])
```

Sum over an axis:

```python
ops.einsum("ij -> i", a)
# array([ 10,  35,  60,  85, 110])
ops.einsum(a, [0, 1], [0])
# array([ 10,  35,  60,  85, 110])
ops.sum(a, axis=1)
# array([ 10,  35,  60,  85, 110])
```

For higher dimensional tensors summing a single axis can be done
with ellipsis:

```python
ops.einsum("...j -> ...", a)
# array([ 10,  35,  60,  85, 110])
np.einsum(a, [..., 1], [...])
# array([ 10,  35,  60,  85, 110])
```

Compute a matrix transpose or reorder any number of axes:

```python
ops.einsum("ji", c)
# array([[0, 3],
#        [1, 4],
#        [2, 5]])
ops.einsum("ij -> ji", c)
# array([[0, 3],
#        [1, 4],
#        [2, 5]])
ops.einsum(c, [1, 0])
# array([[0, 3],
#        [1, 4],
#        [2, 5]])
ops.transpose(c)
# array([[0, 3],
#        [1, 4],
#        [2, 5]])
```

Matrix vector multiplication:

```python
ops.einsum("ij, j", a, b)
# array([ 30,  80, 130, 180, 230])
ops.einsum(a, [0, 1], b, [1])
# array([ 30,  80, 130, 180, 230])
ops.einsum("...j, j", a, b)
# array([ 30,  80, 130, 180, 230])
```

@param subscripts Specifies the subscripts for summation as comma separated
    list of subscript labels. An implicit (classical Einstein
    summation) calculation is performed unless the explicit indicator
    `->` is included as well as subscript labels of the precise
    output form.
@param ... The operands to compute the Einstein sum of.

@export
@family ops
@seealso
+ <https://www.tensorflow.org/api_docs/python/tf/keras/ops/einsum>
