keras.layers.LeakyReLU
__signature__
(negative_slope=0.3, **kwargs)
__doc__
Leaky version of a Rectified Linear Unit activation layer.

This layer allows a small gradient when the unit is not active.

Formula:

``` python
f(x) = alpha * x if x < 0
f(x) = x if x >= 0
```

Example:

``` python
leaky_relu_layer = LeakyReLU(negative_slope=0.5)
input = np.array([-10, -5, 0.0, 5, 10])
result = leaky_relu_layer(input)
# result = [-5. , -2.5,  0. ,  5. , 10.]
```

Args:
    negative_slope: Float >= 0.0. Negative slope coefficient.
      Defaults to `0.3`.
    **kwargs: Base layer keyword arguments, such as
        `name` and `dtype`.
