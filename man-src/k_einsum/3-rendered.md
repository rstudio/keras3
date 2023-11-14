Evaluates the Einstein summation convention on the operands.

@description

# Examples

```r
k_einsum <- keras:::k_einsum
a <- k_arange(25) |> k_reshape(c(5, 5))
b <- k_arange(5)
c <- k_arange(6) |> k_reshape(c(2, 3))
```

Trace of a matrix:


```r
k_einsum("ii", a)
k_trace(a)
```

```
## tf.Tensor(60.0, shape=(), dtype=float64)
## tf.Tensor(60.0, shape=(), dtype=float64)
```

Extract the diagonal:


```r
k_einsum("ii -> i", a)
k_diag(a)
```

```
## tf.Tensor([ 0.  6. 12. 18. 24.], shape=(5), dtype=float64)
## tf.Tensor([ 0.  6. 12. 18. 24.], shape=(5), dtype=float64)
```

Sum over an axis:


```r
k_einsum("ij -> i", a)
k_sum(a, axis = 2)
```

```
## tf.Tensor([ 10.  35.  60.  85. 110.], shape=(5), dtype=float64)
## tf.Tensor([ 10.  35.  60.  85. 110.], shape=(5), dtype=float64)
```

For higher dimensional tensors summing a single axis can be done
with ellipsis:


```r
k_einsum("...j -> ...", a)
k_sum(a, axis = -1)
```

```
## tf.Tensor([ 10.  35.  60.  85. 110.], shape=(5), dtype=float64)
## tf.Tensor([ 10.  35.  60.  85. 110.], shape=(5), dtype=float64)
```

Compute a matrix transpose or reorder any number of axes:


```r
k_einsum("ji", c)
k_einsum("ij -> ji", c)
k_transpose(c)
```

```
## tf.Tensor(
## [[0. 3.]
##  [1. 4.]
##  [2. 5.]], shape=(3, 2), dtype=float64)
## tf.Tensor(
## [[0. 3.]
##  [1. 4.]
##  [2. 5.]], shape=(3, 2), dtype=float64)
## tf.Tensor(
## [[0. 3.]
##  [1. 4.]
##  [2. 5.]], shape=(3, 2), dtype=float64)
```

Matrix vector multiplication:


```r
k_einsum("ij, j", a, b)
k_einsum("...j, j", a, b)
a %*% b
k_matmul(a, b)
```

```
## tf.Tensor([ 30.  80. 130. 180. 230.], shape=(5), dtype=float64)
## tf.Tensor([ 30.  80. 130. 180. 230.], shape=(5), dtype=float64)
## tf.Tensor([ 30.  80. 130. 180. 230.], shape=(5), dtype=float64)
## tf.Tensor([ 30.  80. 130. 180. 230.], shape=(5), dtype=float64)
```

@returns
The calculation based on the Einstein summation convention.

@param subscripts
Specifies the subscripts for summation as comma separated
list of subscript labels. An implicit (classical Einstein
summation) calculation is performed unless the explicit indicator
`->` is included as well as subscript labels of the precise
output form.

@param ...
The operands to compute the Einstein sum of.

@export
@family ops
@seealso
+ <https:/keras.io/keras_core/api/ops/numpy#einsum-function>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/ops/einsum>
