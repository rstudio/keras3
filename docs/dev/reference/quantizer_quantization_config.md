# Base configuration for model quantization

Subclasses provide a quantization `mode` and serialization methods. Use
a concrete configuration such as
[`quantizer_int8_quantization_config()`](https://keras3.posit.co/dev/reference/quantizer_int8_quantization_config.md)
for model quantization. This base class does not define a quantization
mode.

## Usage

``` r
quantizer_quantization_config(
  weight_quantizer = NULL,
  activation_quantizer = NULL
)
```

## Arguments

- weight_quantizer:

  Optional quantizer for weights.

- activation_quantizer:

  Optional quantizer for activations.

## Value

A `QuantizationConfig` instance.

## See also

Other quantizers:  
[`quantizer_abs_max_quantize_grouped_with_zero_point()`](https://keras3.posit.co/dev/reference/quantizer_abs_max_quantize_grouped_with_zero_point.md)  
[`quantizer_awq_config()`](https://keras3.posit.co/dev/reference/quantizer_awq_config.md)  
[`quantizer_float8_quantization_config()`](https://keras3.posit.co/dev/reference/quantizer_float8_quantization_config.md)  
[`quantizer_gptq_config()`](https://keras3.posit.co/dev/reference/quantizer_gptq_config.md)  
[`quantizer_int4_quantization_config()`](https://keras3.posit.co/dev/reference/quantizer_int4_quantization_config.md)  
[`quantizer_int8_quantization_config()`](https://keras3.posit.co/dev/reference/quantizer_int8_quantization_config.md)  
