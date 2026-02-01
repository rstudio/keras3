# Initializer that generates tensors with constant values.

Only scalar values are allowed. The constant value provided must be
convertible to the dtype requested when calling the initializer.

## Usage

``` r
initializer_constant(value = 0)
```

## Arguments

- value:

  A numeric scalar.

## Value

An `Initializer` instance that can be passed to layer or variable
constructors, or called directly with a `shape` to return a Tensor.

## Examples

    # Standalone usage:
    initializer <- initializer_constant(10)
    values <- initializer(shape = c(2, 2))

    # Usage in a Keras layer:
    initializer <- initializer_constant(10)
    layer <- layer_dense(units = 3, kernel_initializer = initializer)

## See also

- <https://keras.io/api/layers/initializers#constant-class>

Other constant initializers:  
[`initializer_identity()`](https://keras3.posit.co/reference/initializer_identity.md)  
[`initializer_ones()`](https://keras3.posit.co/reference/initializer_ones.md)  
[`initializer_stft()`](https://keras3.posit.co/reference/initializer_stft.md)  
[`initializer_zeros()`](https://keras3.posit.co/reference/initializer_zeros.md)  

Other initializers:  
[`initializer_glorot_normal()`](https://keras3.posit.co/reference/initializer_glorot_normal.md)  
[`initializer_glorot_uniform()`](https://keras3.posit.co/reference/initializer_glorot_uniform.md)  
[`initializer_he_normal()`](https://keras3.posit.co/reference/initializer_he_normal.md)  
[`initializer_he_uniform()`](https://keras3.posit.co/reference/initializer_he_uniform.md)  
[`initializer_identity()`](https://keras3.posit.co/reference/initializer_identity.md)  
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
