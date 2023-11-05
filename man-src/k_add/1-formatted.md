Add arguments element-wise.

# Returns
The tensor containing the element-wise sum of `x1` and `x2`.

# Examples
```python
x1 = keras.ops.convert_to_tensor([1, 4])
x2 = keras.ops.convert_to_tensor([5, 6])
keras.ops.add(x1, x2)
# array([6, 10], dtype=int32)
```

`keras.ops.add` also broadcasts shapes:
```python
x1 = keras.ops.convert_to_tensor(
    [[5, 4],
     [5, 6]]
)
x2 = keras.ops.convert_to_tensor([5, 6])
keras.ops.add(x1, x2)
# array([[10 10]
#        [10 12]], shape=(2, 2), dtype=int32)
```

@param x1 First input tensor.
@param x2 Second input tensor.

@export
@family ops
@seealso
+ <https://www.tensorflow.org/api_docs/python/tf/keras/ops/add>
