# Int8 quantization configuration

Configures int8 quantization with an absolute-maximum activation
quantizer by default.

## Usage

``` r
quantizer_int8_quantization_config(
  weight_quantizer = NULL,
  activation_quantizer = "default"
)
```

## Arguments

- weight_quantizer:

  Optional quantizer for weights.

- activation_quantizer:

  Optional quantizer for activations. The default uses an
  absolute-maximum quantizer over the last axis.

## Value

An `Int8QuantizationConfig` instance.

## Examples

    config <- quantizer_int8_quantization_config()

## See also

Other quantizers:  
[`quantizer_abs_max_quantize_grouped_with_zero_point()`](https://keras3.posit.co/dev/reference/quantizer_abs_max_quantize_grouped_with_zero_point.md)  
[`quantizer_awq_config()`](https://keras3.posit.co/dev/reference/quantizer_awq_config.md)  
[`quantizer_float8_quantization_config()`](https://keras3.posit.co/dev/reference/quantizer_float8_quantization_config.md)  
[`quantizer_gptq_config()`](https://keras3.posit.co/dev/reference/quantizer_gptq_config.md)  
[`quantizer_int4_quantization_config()`](https://keras3.posit.co/dev/reference/quantizer_int4_quantization_config.md)  
[`quantizer_quantization_config()`](https://keras3.posit.co/dev/reference/quantizer_quantization_config.md)  
