Leaky version of a Rectified Linear Unit activation function.

@description
It allows a small gradient when the unit is not active, it is defined as:

`f(x) = alpha * x for x < 0` or `f(x) = x for x >= 0`.

# Returns
A tensor with the same shape as `x`.

# Examples
```python
x = np.array([-1., 0., 1.])
x_leaky_relu = keras.ops.leaky_relu(x)
print(x_leaky_relu)
# array([-0.2,  0. ,  1. ], shape=(3,), dtype=float64)
```

@param x Input tensor.
@param negative_slope Slope of the activation function at x < 0.
    Defaults to `0.2`.

@export
@family ops
@seealso
+ <https://www.tensorflow.org/api_docs/python/tf/keras/ops/leaky_relu>
