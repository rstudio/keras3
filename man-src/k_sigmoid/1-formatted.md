Sigmoid activation function.

@description
It is defined as `f(x) = 1 / (1 + exp(-x))`.

# Examples
```python
x = keras.ops.convert_to_tensor([-6.0, 1.0, 0.0, 1.0, 6.0])
keras.ops.sigmoid(x)
# array([0.00247262, 0.7310586, 0.5, 0.7310586, 0.9975274], dtype=float32)
```

@returns
A tensor with the same shape as `x`.

@param x
Input tensor.

@export
@family nn ops
@family ops
@seealso
+ <https:/keras.io/keras_core/api/ops/nn#sigmoid-function>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/ops/sigmoid>
