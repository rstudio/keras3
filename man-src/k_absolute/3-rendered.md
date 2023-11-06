Compute the absolute value element-wise.

@description
`keras::k_abs` is a shorthand for this function.

# Examples

```r
x <- k_constant(c(-1.2, 1.2))
k_abs(x)
```

```
## tf.Tensor([1.2 1.2], shape=(2), dtype=float64)
```

@returns
A tensor containing the absolute value of each element in `x`.

@param x Input tensor.

@export
@family ops
@seealso
+ <https://www.tensorflow.org/api_docs/python/tf/keras/ops/absolute>
