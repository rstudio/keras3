Cast a tensor to the desired dtype.

@description

# Examples

```r
(x <- k_arange(4))
```

```
## tf.Tensor([0 1 2 3], shape=(4), dtype=int32)
```

```r
k_cast(x, dtype = "float16")
```

```
## tf.Tensor([0. 1. 2. 3.], shape=(4), dtype=float16)
```

@returns
A tensor of the specified `dtype`.

@param x A tensor or variable.
@param dtype The target type.

@export
@family ops
@seealso
+ <https://www.tensorflow.org/api_docs/python/tf/keras/ops/cast>
