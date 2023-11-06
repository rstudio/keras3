Trigonometric inverse cosine, element-wise.

@description
The inverse of `cos` so that, if `y = cos(x)`, then `x = arccos(y)`.

# Examples

```r
x <- k_convert_to_tensor(c(1, -1))
k_arccos(x)
```

```
## tf.Tensor([0.        3.1415925], shape=(2), dtype=float32)
```

@returns
Tensor of the angle of the ray intersecting the unit circle at the given
x-coordinate in radians `[0, pi]`.

@param x Input tensor.

@export
@family ops
@seealso
+ <https://www.tensorflow.org/api_docs/python/tf/keras/ops/arccos>
