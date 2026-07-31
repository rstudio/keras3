# Grouped asymmetric absolute-maximum quantization

Quantizes a 2D tensor in row groups. Each group receives a scale and
zero point for every output column.

## Usage

``` r
quantizer_abs_max_quantize_grouped_with_zero_point(
  inputs,
  block_size,
  value_range = c(-8L, 7L),
  dtype = "int8",
  epsilon = 1e-07,
  to_numpy = FALSE
)
```

## Arguments

- inputs:

  A 2D tensor with shape `(input_dim, output_dim)`.

- block_size:

  Number of rows in each quantization group.

- value_range:

  Integer vector giving the minimum and maximum quantized values.

- dtype:

  Dtype of the quantized output.

- epsilon:

  Small value used to avoid division by zero.

- to_numpy:

  Whether to perform the computation in NumPy for memory efficiency.

## Value

A list containing the quantized tensor, group scales, and group zero
points.

## Examples

    kernel <- op_reshape(op_arange(16, dtype = "float32"), c(4, 4))
    result <- quantizer_abs_max_quantize_grouped_with_zero_point(
      kernel,
      block_size = 2
    )

## See also

Other quantizers:  
[`quantizer_awq_config()`](https://keras3.posit.co/dev/reference/quantizer_awq_config.md)  
[`quantizer_float8_quantization_config()`](https://keras3.posit.co/dev/reference/quantizer_float8_quantization_config.md)  
[`quantizer_gptq_config()`](https://keras3.posit.co/dev/reference/quantizer_gptq_config.md)  
[`quantizer_int4_quantization_config()`](https://keras3.posit.co/dev/reference/quantizer_int4_quantization_config.md)  
[`quantizer_int8_quantization_config()`](https://keras3.posit.co/dev/reference/quantizer_int8_quantization_config.md)  
[`quantizer_quantization_config()`](https://keras3.posit.co/dev/reference/quantizer_quantization_config.md)  
