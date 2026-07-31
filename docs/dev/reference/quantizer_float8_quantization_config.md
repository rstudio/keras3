# Float8 quantization configuration

Float8 mixed-precision training does not support user-defined
quantizers; this object selects the built-in float8 behavior.

## Usage

``` r
quantizer_float8_quantization_config()
```

## Value

A `Float8QuantizationConfig` instance.

## Examples

    config <- quantizer_float8_quantization_config()

## See also

Other quantizers:  
[`quantizer_abs_max_quantize_grouped_with_zero_point()`](https://keras3.posit.co/dev/reference/quantizer_abs_max_quantize_grouped_with_zero_point.md)  
[`quantizer_awq_config()`](https://keras3.posit.co/dev/reference/quantizer_awq_config.md)  
[`quantizer_gptq_config()`](https://keras3.posit.co/dev/reference/quantizer_gptq_config.md)  
[`quantizer_int4_quantization_config()`](https://keras3.posit.co/dev/reference/quantizer_int4_quantization_config.md)  
[`quantizer_int8_quantization_config()`](https://keras3.posit.co/dev/reference/quantizer_int8_quantization_config.md)  
[`quantizer_quantization_config()`](https://keras3.posit.co/dev/reference/quantizer_quantization_config.md)  
