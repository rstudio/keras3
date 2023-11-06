Softplus activation function.

@description
It is defined as `f(x) = log(exp(x) + 1)`, where `log` is the natural
logarithm and `exp` is the exponential function.

# Examples
```python
x = keras.ops.convert_to_tensor([-0.555, 0.0, 0.555])
keras.ops.softplus(x)
# array([0.45366603, 0.6931472, 1.008666], dtype=float32)
```

@returns
A tensor with the same shape as `x`.

@param x Input tensor.

@export
@family ops
@seealso
+ <https://www.tensorflow.org/api_docs/python/tf/keras/ops/softplus>
