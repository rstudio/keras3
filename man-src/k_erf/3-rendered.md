Computes the error function of `x`, element-wise.

@description

# Examples

```r
x <- k_array(c(-3, -2, -1, 0, 1))
k_erf(x)
```

```
## tf.Tensor([-0.99997795 -0.9953222  -0.84270084  0.          0.84270084], shape=(5), dtype=float32)
```

```r
# array([-0.99998 , -0.99532, -0.842701,  0.,  0.842701], dtype=float32)
```

@returns
A tensor with the same dtype as `x`.

@param x
Input tensor.

@export
@family math ops
@family ops
@seealso
+ <https://www.tensorflow.org/api_docs/python/tf/keras/ops/erf>
