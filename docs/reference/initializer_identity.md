# Initializer that generates the identity matrix.

Only usable for generating 2D matrices.

## Usage

``` r
initializer_identity(gain = 1)
```

## Arguments

- gain:

  Multiplicative factor to apply to the identity matrix.

## Value

An `Initializer` instance that can be passed to layer or variable
constructors, or called directly with a `shape` to return a Tensor.

## Examples

    # Standalone usage:
    initializer <- initializer_identity()
    values <- initializer(shape = c(2, 2))

    # Usage in a Keras layer:
    initializer <- initializer_identity()
    layer <- layer_dense(units = 3, kernel_initializer = initializer)

## See also

Other constant initializers:  
[`initializer_constant()`](https://keras3.posit.co/reference/initializer_constant.md)  
[`initializer_ones()`](https://keras3.posit.co/reference/initializer_ones.md)  
[`initializer_stft()`](https://keras3.posit.co/reference/initializer_stft.md)  
[`initializer_zeros()`](https://keras3.posit.co/reference/initializer_zeros.md)  

Other initializers:  
[`initializer_constant()`](https://keras3.posit.co/reference/initializer_constant.md)  
[`initializer_glorot_normal()`](https://keras3.posit.co/reference/initializer_glorot_normal.md)  
[`initializer_glorot_uniform()`](https://keras3.posit.co/reference/initializer_glorot_uniform.md)  
[`initializer_he_normal()`](https://keras3.posit.co/reference/initializer_he_normal.md)  
[`initializer_he_uniform()`](https://keras3.posit.co/reference/initializer_he_uniform.md)  
[`initializer_lecun_normal()`](https://keras3.posit.co/reference/initializer_lecun_normal.md)  
[`initializer_lecun_uniform()`](https://keras3.posit.co/reference/initializer_lecun_uniform.md)  
[`initializer_ones()`](https://keras3.posit.co/reference/initializer_ones.md)  
[`initializer_orthogonal()`](https://keras3.posit.co/reference/initializer_orthogonal.md)  
[`initializer_random_normal()`](https://keras3.posit.co/reference/initializer_random_normal.md)  
[`initializer_random_uniform()`](https://keras3.posit.co/reference/initializer_random_uniform.md)  
[`initializer_stft()`](https://keras3.posit.co/reference/initializer_stft.md)  
[`initializer_truncated_normal()`](https://keras3.posit.co/reference/initializer_truncated_normal.md)  
[`initializer_variance_scaling()`](https://keras3.posit.co/reference/initializer_variance_scaling.md)  
[`initializer_zeros()`](https://keras3.posit.co/reference/initializer_zeros.md)  
