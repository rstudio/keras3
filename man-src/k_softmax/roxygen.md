Softmax activation function.

@description
The elements of the output vector lie within the range `(0, 1)`, and their
total sum is exactly 1 (excluding the floating point rounding error).

Each vector is processed independently. The `axis` argument specifies the
axis along which the function is applied within the input.

It is defined as:
`f(x) = exp(x) / sum(exp(x))`

# Returns
A tensor with the same shape as `x`.

# Examples
```python
x = np.array([-1., 0., 1.])
x_softmax = keras.ops.softmax(x)
print(x_softmax)
# array([0.09003057, 0.24472847, 0.66524096], shape=(3,), dtype=float64)
```

@param x Input tensor.
@param axis Integer, axis along which the softmax is applied.

@export
@family ops
@seealso
+ <https://www.tensorflow.org/api_docs/python/tf/keras/ops/softmax>
