keras.layers.ReLU
__signature__
(
  max_value=None,
  negative_slope=0.0,
  threshold=0.0,
  **kwargs
)
__doc__
Rectified Linear Unit activation function layer.

Formula:
``` python
f(x) = max(x,0)
f(x) = max_value if x >= max_value
f(x) = x if threshold <= x < max_value
f(x) = negative_slope * (x - threshold) otherwise
```

Example:
``` python
relu_layer = keras.layers.activations.ReLU(
    max_value=10,
    negative_slope=0.5,
    threshold=0,
)
input = np.array([-10, -5, 0.0, 5, 10])
result = relu_layer(input)
# result = [-5. , -2.5,  0. ,  5. , 10.]
```

Args:
    max_value: Float >= 0. Maximum activation value. None means unlimited.
        Defaults to `None`.
    negative_slope: Float >= 0. Negative slope coefficient.
        Defaults to `0.0`.
    threshold: Float >= 0. Threshold value for thresholded activation.
        Defaults to `0.0`.
    **kwargs: Base layer keyword arguments, such as `name` and `dtype`.
