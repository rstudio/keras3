Computes element-wise dot product of two tensors.

@description
It takes a list of inputs of size 2, and the axes
corresponding to each input along with the dot product
is to be performed.

Let's say `x` and `y` are the two input tensors with shapes
`(2, 3, 5)` and `(2, 10, 3)`. The batch dimension should be
of same size for both the inputs, and `axes` should correspond
to the dimensions that have the same size in the corresponding
inputs. e.g. with `axes = c(1, 2)`, the dot product of `x`, and `y`
will result in a tensor with shape `(2, 5, 10)`

# Examples

```r
x <- k_reshape(0:9,   c(1, 5, 2))
y <- k_reshape(10:19, c(1, 2, 5))
layer_dot(x, y, axes=c(2, 3))
```

```
## tf.Tensor(
## [[[260 360]
##   [320 445]]], shape=(1, 2, 2), dtype=int64)
```

Usage in a Keras model:


```r
x1 <- k_reshape(0:9, c(5, 2)) |> layer_dense(8)
x2 <- k_reshape(10:19, c(5, 2)) |> layer_dense(8)
shape(x1)
```

```
## shape(5, 8)
```

```r
shape(x2)
```

```
## shape(5, 8)
```

```r
y <- layer_dot(x1, x2, axes=2)
shape(y)
```

```
## shape(5, 1)
```

@returns
    A tensor, the dot product of the samples from the inputs.

@param axes
Integer or list of integers, axis or axes along which to
take the dot product. If a list, should be two integers
corresponding to the desired axis from the first input and the
desired axis from the second input, respectively. Note that the
size of the two selected axes must match.

@param normalize
Whether to L2-normalize samples along the dot product axis
before taking the dot product. If set to `TRUE`, then
the output of the dot product is the cosine proximity
between the two samples.

@param ...
Standard layer keyword arguments.

@param inputs
layers to combine

@export
@family merging layers
@family layers
@seealso
+ <https:/keras.io/api/layers/merging_layers/dot#dot-class>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dot>
