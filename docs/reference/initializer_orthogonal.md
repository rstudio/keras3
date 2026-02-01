# Initializer that generates an orthogonal matrix.

If the shape of the tensor to initialize is two-dimensional, it is
initialized with an orthogonal matrix obtained from the QR decomposition
of a matrix of random numbers drawn from a normal distribution. If the
matrix has fewer rows than columns then the output will have orthogonal
rows. Otherwise, the output will have orthogonal columns.

If the shape of the tensor to initialize is more than two-dimensional, a
matrix of shape `(shape[1] * ... * shape[n - 1], shape[n])` is
initialized, where `n` is the length of the shape vector. The matrix is
subsequently reshaped to give a tensor of the desired shape.

## Usage

``` r
initializer_orthogonal(gain = 1, seed = NULL)
```

## Arguments

- gain:

  Multiplicative factor to apply to the orthogonal matrix.

- seed:

  An integer. Used to make the behavior of the initializer
  deterministic.

## Value

An `Initializer` instance that can be passed to layer or variable
constructors, or called directly with a `shape` to return a Tensor.

## Examples

    # Standalone usage:
    initializer <- initializer_orthogonal()
    values <- initializer(shape = c(2, 2))

    # Usage in a Keras layer:
    initializer <- initializer_orthogonal()
    layer <- layer_dense(units = 3, kernel_initializer = initializer)

## Reference

- [Saxe et al., 2014](https://openreview.net/forum?id=_wzZwKpTDF_9C)

## See also

Other random initializers:  
[`initializer_glorot_normal()`](https://keras3.posit.co/reference/initializer_glorot_normal.md)  
[`initializer_glorot_uniform()`](https://keras3.posit.co/reference/initializer_glorot_uniform.md)  
[`initializer_he_normal()`](https://keras3.posit.co/reference/initializer_he_normal.md)  
[`initializer_he_uniform()`](https://keras3.posit.co/reference/initializer_he_uniform.md)  
[`initializer_lecun_normal()`](https://keras3.posit.co/reference/initializer_lecun_normal.md)  
[`initializer_lecun_uniform()`](https://keras3.posit.co/reference/initializer_lecun_uniform.md)  
[`initializer_random_normal()`](https://keras3.posit.co/reference/initializer_random_normal.md)  
[`initializer_random_uniform()`](https://keras3.posit.co/reference/initializer_random_uniform.md)  
[`initializer_truncated_normal()`](https://keras3.posit.co/reference/initializer_truncated_normal.md)  
[`initializer_variance_scaling()`](https://keras3.posit.co/reference/initializer_variance_scaling.md)  

Other initializers:  
[`initializer_constant()`](https://keras3.posit.co/reference/initializer_constant.md)  
[`initializer_glorot_normal()`](https://keras3.posit.co/reference/initializer_glorot_normal.md)  
[`initializer_glorot_uniform()`](https://keras3.posit.co/reference/initializer_glorot_uniform.md)  
[`initializer_he_normal()`](https://keras3.posit.co/reference/initializer_he_normal.md)  
[`initializer_he_uniform()`](https://keras3.posit.co/reference/initializer_he_uniform.md)  
[`initializer_identity()`](https://keras3.posit.co/reference/initializer_identity.md)  
[`initializer_lecun_normal()`](https://keras3.posit.co/reference/initializer_lecun_normal.md)  
[`initializer_lecun_uniform()`](https://keras3.posit.co/reference/initializer_lecun_uniform.md)  
[`initializer_ones()`](https://keras3.posit.co/reference/initializer_ones.md)  
[`initializer_random_normal()`](https://keras3.posit.co/reference/initializer_random_normal.md)  
[`initializer_random_uniform()`](https://keras3.posit.co/reference/initializer_random_uniform.md)  
[`initializer_stft()`](https://keras3.posit.co/reference/initializer_stft.md)  
[`initializer_truncated_normal()`](https://keras3.posit.co/reference/initializer_truncated_normal.md)  
[`initializer_variance_scaling()`](https://keras3.posit.co/reference/initializer_variance_scaling.md)  
[`initializer_zeros()`](https://keras3.posit.co/reference/initializer_zeros.md)  
