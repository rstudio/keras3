# Int4 quantization configuration

Configures weight-only int4 quantization by default, using groups along
the input dimension.

## Usage

``` r
quantizer_int4_quantization_config(
  weight_quantizer = NULL,
  activation_quantizer = "default",
  block_size = 128L
)
```

## Arguments

- weight_quantizer:

  Optional quantizer for weights.

- activation_quantizer:

  Optional quantizer for activations. The default selects weight-only
  int4 quantization.

- block_size:

  Group size along the input dimension. A positive integer uses
  sub-channel quantization with `ceiling(input_dim / block_size)`
  groups; `NULL` or `-1` uses per-channel quantization with one scale
  per output channel. Sub-channel quantization does not support custom
  weight or activation quantizers. Defaults to 128.

## Value

An `Int4QuantizationConfig` instance.

## Examples

    config <- quantizer_int4_quantization_config(block_size = 128)

## See also

Other quantizers:  
[`quantizer_abs_max_quantize_grouped_with_zero_point()`](https://keras3.posit.co/dev/reference/quantizer_abs_max_quantize_grouped_with_zero_point.md)  
[`quantizer_awq_config()`](https://keras3.posit.co/dev/reference/quantizer_awq_config.md)  
[`quantizer_float8_quantization_config()`](https://keras3.posit.co/dev/reference/quantizer_float8_quantization_config.md)  
[`quantizer_gptq_config()`](https://keras3.posit.co/dev/reference/quantizer_gptq_config.md)  
[`quantizer_int8_quantization_config()`](https://keras3.posit.co/dev/reference/quantizer_int8_quantization_config.md)  
[`quantizer_quantization_config()`](https://keras3.posit.co/dev/reference/quantizer_quantization_config.md)  
