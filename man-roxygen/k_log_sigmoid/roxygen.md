Logarithm of the sigmoid activation function.

@description
It is defined as `f(x) = log(1 / (1 + exp(-x)))`.

# Returns
A tensor with the same shape as `x`.

# Examples
```python
x = keras.ops.convert_to_tensor([-0.541391, 0.0, 0.50, 5.0])
keras.ops.log_sigmoid(x)
# array([-1.0000418, -0.6931472, -0.474077, -0.00671535], dtype=float32)
```

@param x Input tensor.

@export
@family ops
@seealso
+ <https://www.tensorflow.org/api_docs/python/tf/keras/ops/log_sigmoid>
