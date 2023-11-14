Element-wise arc tangent of `x1/x2` choosing the quadrant correctly.

@description
The quadrant (i.e., branch) is chosen so that `arctan2(x1, x2)` is the
signed angle in radians between the ray ending at the origin and passing
through the point `(1, 0)`, and the ray ending at the origin and passing
through the point `(x2, x1)`. (Note the role reversal: the "y-coordinate"
is the first function parameter, the "x-coordinate" is the second.) By IEEE
convention, this function is defined for `x2 = +/-0` and for either or both
of `x1` and `x2` `= +/-inf`.

# Examples
Consider four points in different quadrants:

```r
x <- k_array(c(-1, 1, 1, -1))
y <- k_array(c(-1, -1, 1, 1))
k_arctan2(y, x) * 180 / pi
```

```
## tf.Tensor([-135.        -44.999996   44.999996  135.      ], shape=(4), dtype=float32)
```

Note the order of the parameters. `arctan2` is defined also when x2 = 0 and
at several other points, obtaining values in the range `[-pi, pi]`:

```r
k_arctan2(
    k_array(c(1, -1)),
    k_array(c(0, 0))
)
```

```
## tf.Tensor([ 1.5707964 -1.5707964], shape=(2), dtype=float32)
```

```r
k_arctan2(
    k_array(c(0, 0, Inf)),
    k_array(c(+0, -0, Inf))
)
```

```
## tf.Tensor([0.        3.1415925 0.7853982], shape=(3), dtype=float32)
```

@returns
Tensor of angles in radians, in the range `[-pi, pi]`.

@param x1
First input tensor.

@param x2
Second input tensor.

@export
@family numpy ops
@family ops
@seealso
+ <https:/keras.io/keras_core/api/ops/numpy#arctan2-function>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/ops/arctan2>
