Cast a tensor to the desired dtype.

@description

# Examples
```python
x = keras.ops.arange(4)
x = keras.ops.cast(x, dtype="float16")
```

@returns
A tensor of the specified `dtype`.

@param x A tensor or variable.
@param dtype The target type.

@export
@family ops
@seealso
+ <https://www.tensorflow.org/api_docs/python/tf/keras/ops/cast>
