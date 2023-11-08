keras.layers.ELU
__signature__
(alpha=1.0, **kwargs)
__doc__
Applies an Exponential Linear Unit function to an output.

Formula:

```
f(x) = alpha * (exp(x) - 1.) for x < 0
f(x) = x for x >= 0
```

Args:
    alpha: float, slope of negative section. Defaults to `1.0`.
    **kwargs: Base layer keyword arguments, such as `name` and `dtype`.
