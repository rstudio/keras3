Calculate the n-th discrete difference along the given axis.

@description
The first difference is given by `out[i] = a[i+1] - a[i]` along
the given axis, higher differences are calculated by using `diff`
recursively.

# Examples

```r
x <- k_array(c(1, 2, 4, 7, 0))
k_diff(x)
```

```
## tf.Tensor([ 1.  2.  3. -7.], shape=(4), dtype=float32)
```

```r
k_diff(x, n = 2)
```

```
## tf.Tensor([  1.   1. -10.], shape=(3), dtype=float32)
```

```r
x <- k_array(rbind(c(1, 3, 6, 10),
                  c(0, 5, 6, 8)))
k_diff(x)
```

```
## tf.Tensor(
## [[2. 3. 4.]
##  [5. 1. 2.]], shape=(2, 3), dtype=float64)
```

```r
k_diff(x, axis = 1)
```

```
## tf.Tensor([[-1.  2.  0. -2.]], shape=(1, 4), dtype=float64)
```

@returns
Tensor of diagonals.

@param a
Input tensor.

@param n
The number of times values are differenced. Defaults to `1`.

@param axis
Axis to compute discrete difference(s) along.
Defaults to `-1` (last axis).

@export
@family numpy ops
@family ops
@seealso
+ <https://www.tensorflow.org/api_docs/python/tf/keras/ops/diff>
