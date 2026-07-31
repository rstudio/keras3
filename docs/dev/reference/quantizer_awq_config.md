# AWQ calibration and quantization configuration

Activation-aware Weight Quantization uses calibration activations to
find per-channel scales that protect salient weights.

## Usage

``` r
quantizer_awq_config(
  dataset,
  tokenizer,
  ...,
  weight_bits = 4L,
  num_samples = 128L,
  sequence_length = 512L,
  group_size = 128L,
  num_grid_points = 20L,
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

  Number of weight bits. AWQ currently supports 4.

- num_samples:

  Number of calibration samples.

- sequence_length:

  Sequence length of each calibration sample.

- group_size:

  Number of weights quantized together. Use `-1` for per-channel
  quantization.

- num_grid_points:

  Number of grid-search points used to find scales.

- quantization_layer_structure:

  Optional named list describing `pre_block_layers` and
  `sequential_blocks`. If omitted, the model must provide
  `get_quantization_layer_structure()`.

## Value

An `AWQConfig` instance.

## Examples

    config <- quantizer_awq_config(
      dataset = calibration_text,
      tokenizer = tokenizer,
      group_size = 128
    )
    model |> quantize_weights(mode = "awq", config = config)

## See also

Other quantizers:  
[`quantizer_abs_max_quantize_grouped_with_zero_point()`](https://keras3.posit.co/dev/reference/quantizer_abs_max_quantize_grouped_with_zero_point.md)  
[`quantizer_float8_quantization_config()`](https://keras3.posit.co/dev/reference/quantizer_float8_quantization_config.md)  
[`quantizer_gptq_config()`](https://keras3.posit.co/dev/reference/quantizer_gptq_config.md)  
[`quantizer_int4_quantization_config()`](https://keras3.posit.co/dev/reference/quantizer_int4_quantization_config.md)  
[`quantizer_int8_quantization_config()`](https://keras3.posit.co/dev/reference/quantizer_int8_quantization_config.md)  
[`quantizer_quantization_config()`](https://keras3.posit.co/dev/reference/quantizer_quantization_config.md)  
