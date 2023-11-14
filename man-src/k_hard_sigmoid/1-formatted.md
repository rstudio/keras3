Hard sigmoid activation function.

@description
It is defined as:

`0 if x < -2.5`, `1 if x > 2.5`, `(0.2 * x) + 0.5 if -2.5 <= x <= 2.5`.

# Examples
```python
x = np.array([-1., 0., 1.])
x_hard_sigmoid = keras.ops.hard_sigmoid(x)
print(x_hard_sigmoid)
# array([0.3, 0.5, 0.7], shape=(3,), dtype=float64)
```

@returns
A tensor with the same shape as `x`.

@param x
Input tensor.

@export
@family ops
@seealso
+ <https:/keras.io/keras_core/api/ops/nn#hardsigmoid-function>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/ops/hard_sigmoid>
