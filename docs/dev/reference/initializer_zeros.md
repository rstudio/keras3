# Initializer that generates tensors initialized to 0.

Initializer that generates tensors initialized to 0.

## Usage

``` r
initializer_zeros()
```

## Value

An `Initializer` instance that can be passed to layer or variable
constructors, or called directly with a `shape` to return a Tensor.

## Examples

    # Standalone usage:
    initializer <- initializer_zeros()
    values <- initializer(shape = c(2, 2))

    # Usage in a Keras layer:
    initializer <- initializer_zeros()
    layer <- layer_dense(units = 3, kernel_initializer = initializer)

## See also

- <https://keras.io/api/layers/initializers#zeros-class>

Other constant initializers:  
[`initializer_constant()`](https://keras3.posit.co/dev/reference/initializer_constant.md)  
[`initializer_identity()`](https://keras3.posit.co/dev/reference/initializer_identity.md)  
[`initializer_ones()`](https://keras3.posit.co/dev/reference/initializer_ones.md)  
[`initializer_stft()`](https://keras3.posit.co/dev/reference/initializer_stft.md)  

Other initializers:  
[`initializer_constant()`](https://keras3.posit.co/dev/reference/initializer_constant.md)  
[`initializer_glorot_normal()`](https://keras3.posit.co/dev/reference/initializer_glorot_normal.md)  
[`initializer_glorot_uniform()`](https://keras3.posit.co/dev/reference/initializer_glorot_uniform.md)  
[`initializer_he_normal()`](https://keras3.posit.co/dev/reference/initializer_he_normal.md)  
[`initializer_he_uniform()`](https://keras3.posit.co/dev/reference/initializer_he_uniform.md)  
[`initializer_identity()`](https://keras3.posit.co/dev/reference/initializer_identity.md)  
[`initializer_lecun_normal()`](https://keras3.posit.co/dev/reference/initializer_lecun_normal.md)  
[`initializer_lecun_uniform()`](https://keras3.posit.co/dev/reference/initializer_lecun_uniform.md)  
[`initializer_ones()`](https://keras3.posit.co/dev/reference/initializer_ones.md)  
[`initializer_orthogonal()`](https://keras3.posit.co/dev/reference/initializer_orthogonal.md)  
[`initializer_random_normal()`](https://keras3.posit.co/dev/reference/initializer_random_normal.md)  
[`initializer_random_uniform()`](https://keras3.posit.co/dev/reference/initializer_random_uniform.md)  
[`initializer_stft()`](https://keras3.posit.co/dev/reference/initializer_stft.md)  
[`initializer_truncated_normal()`](https://keras3.posit.co/dev/reference/initializer_truncated_normal.md)  
[`initializer_variance_scaling()`](https://keras3.posit.co/dev/reference/initializer_variance_scaling.md)  
