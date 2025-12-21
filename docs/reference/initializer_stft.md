# Initializer of Conv kernels for Short-term Fourier Transformation (STFT).

Since the formula involves complex numbers, this class compute either
the real or the imaginary components of the final output.

Additionally, this initializer supports windowing functions across the
time dimension as commonly used in STFT. Windowing functions from the
Python module `scipy.signal.windows` are supported, including the common
`hann` and `hamming` windowing functions. This layer supports periodic
windows and scaling-based normalization.

This is primarily intended for use in the `STFTSpectrogram` layer.

## Usage

``` r
initializer_stft(
  side = "real",
  window = "hann",
  scaling = "density",
  periodic = FALSE
)
```

## Arguments

- side:

  String, `"real"` or `"imag"` deciding if the kernel will compute the
  real side or the imaginary side of the output. Defaults to `"real"`.

- window:

  String for the name of the windowing function in the
  `scipy.signal.windows` module, or array_like for the window values, or
  `NULL` for no windowing.

- scaling:

  String, `"density"` or `"spectrum"` for scaling of the window for
  normalization, either L2 or L1 normalization. `NULL` for no scaling.

- periodic:

  Boolean, if True, the window function will be treated as periodic.
  Defaults to `FALSE`.

## Value

An `Initializer` instance that can be passed to layer or variable
constructors, or called directly with a `shape` to return a Tensor.

## Examples

    # Standalone usage:
    initializer <- initializer_stft("real", "hann", "density", FALSE)
    values <- initializer(shape = c(128, 1, 513))

## See also

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
[`initializer_orthogonal()`](https://keras3.posit.co/reference/initializer_orthogonal.md)  
[`initializer_random_normal()`](https://keras3.posit.co/reference/initializer_random_normal.md)  
[`initializer_random_uniform()`](https://keras3.posit.co/reference/initializer_random_uniform.md)  
[`initializer_truncated_normal()`](https://keras3.posit.co/reference/initializer_truncated_normal.md)  
[`initializer_variance_scaling()`](https://keras3.posit.co/reference/initializer_variance_scaling.md)  
[`initializer_zeros()`](https://keras3.posit.co/reference/initializer_zeros.md)  

Other constant initializers:  
[`initializer_constant()`](https://keras3.posit.co/reference/initializer_constant.md)  
[`initializer_identity()`](https://keras3.posit.co/reference/initializer_identity.md)  
[`initializer_ones()`](https://keras3.posit.co/reference/initializer_ones.md)  
[`initializer_zeros()`](https://keras3.posit.co/reference/initializer_zeros.md)  
