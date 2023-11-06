Initializer that generates an orthogonal matrix.

@description
If the shape of the tensor to initialize is two-dimensional, it is
initialized with an orthogonal matrix obtained from the QR decomposition of
a matrix of random numbers drawn from a normal distribution. If the matrix
has fewer rows than columns then the output will have orthogonal rows.
Otherwise, the output will have orthogonal columns.

If the shape of the tensor to initialize is more than two-dimensional,
a matrix of shape `(shape[1] * ... * shape[n - 1], shape[n])`
is initialized, where `n` is the length of the shape vector.
The matrix is subsequently reshaped to give a tensor of the desired shape.

# Examples

```r
# Standalone usage:
initializer <- initializer_orthogonal()
values <- initializer(shape = c(2, 2))
```


```r
# Usage in a Keras layer:
initializer <- initializer_orthogonal()
layer <- layer_dense(units = 3, kernel_initializer = initializer)
```

# Reference
- [Saxe et al., 2014](https://openreview.net/forum?id=_wzZwKpTDF_9C)

@param gain Multiplicative factor to apply to the orthogonal matrix.
@param seed An integer. Used to make the behavior of the initializer
    deterministic.

@export
@family initializer
@seealso
+ <https://www.tensorflow.org/api_docs/python/tf/keras/initializers/OrthogonalInitializer>
