# GPTQ calibration and quantization configuration

GPTQ uses calibration data and a Hessian approximation to quantize
weights while reducing quantization error.

## Usage

``` r
quantizer_gptq_config(
  dataset,
  tokenizer,
  ...,
  weight_bits = 4L,
  num_samples = 128L,
  per_channel = TRUE,
  sequence_length = 512L,
  hessian_damping = 0.01,
  group_size = 128L,
  symmetric = FALSE,
  activation_order = FALSE,
  quantization_layer_structure = NULL
)
```

## Arguments

- dataset:

  Calibration data yielding strings or pre-tokenized tensors.

- tokenizer:

  Tokenizer or compatible callable used for `dataset`.

- ...:

  For forward/backward compatibility. Arguments after `...` must be
  named.

- weight_bits:

  Number of weight bits. GPTQ supports 2, 3, 4, and 8.

- num_samples:

  Number of calibration samples.

- per_channel:

  Whether to calculate quantization parameters per output channel.

- sequence_length:

  Sequence length of each calibration sample.

- hessian_damping:

  Fraction of Hessian damping used for numerical stability. Must be
  between 0 and 1.

- group_size:

  Number of weights quantized together. Use `-1` for per-channel
  quantization.

- symmetric:

  Whether to use symmetric quantization.

- activation_order:

  Whether to reorder weight columns by activation magnitude.

- quantization_layer_structure:

  Optional named list describing `pre_block_layers` and
  `sequential_blocks`. If omitted, the model must provide
  `get_quantization_layer_structure()`.

## Value

A `GPTQConfig` instance.

## Examples

    config <- quantizer_gptq_config(
      dataset = calibration_text,
      tokenizer = tokenizer,
      weight_bits = 4,
      group_size = 128
    )
    model |> quantize_weights(mode = "gptq", config = config)

## See also

Other quantizers:  
[`quantizer_abs_max_quantize_grouped_with_zero_point()`](https://keras3.posit.co/dev/reference/quantizer_abs_max_quantize_grouped_with_zero_point.md)  
[`quantizer_awq_config()`](https://keras3.posit.co/dev/reference/quantizer_awq_config.md)  
[`quantizer_float8_quantization_config()`](https://keras3.posit.co/dev/reference/quantizer_float8_quantization_config.md)  
[`quantizer_int4_quantization_config()`](https://keras3.posit.co/dev/reference/quantizer_int4_quantization_config.md)  
[`quantizer_int8_quantization_config()`](https://keras3.posit.co/dev/reference/quantizer_int8_quantization_config.md)  
[`quantizer_quantization_config()`](https://keras3.posit.co/dev/reference/quantizer_quantization_config.md)  
