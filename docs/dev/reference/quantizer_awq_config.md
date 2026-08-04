# AWQ calibration and quantization configuration

Activation-aware Weight Quantization is a post-training method that
identifies and protects salient weights based on activation magnitudes.
It uses calibration data to collect activation statistics, identifies
salient weight channels, searches a grid for optimal per-channel scales,
and applies those scales before quantization to reduce accuracy loss.

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

  Calibration data used to analyze activation patterns. It can be an
  iterable that yields strings or pre-tokenized numeric tensors, such as
  a character vector, generator, R array, or NumPy array.

- tokenizer:

  Tokenizer or compatible callable used to process `dataset` when it
  contains strings.

- ...:

  For forward/backward compatibility. Arguments after `...` must be
  named.

- weight_bits:

  Number of bits used for weight quantization. AWQ currently supports
  only 4. Defaults to 4.

- num_samples:

  Number of calibration samples to use. Defaults to 128.

- sequence_length:

  Sequence length of each calibration sample. Defaults to 512.

- group_size:

  Number of weights quantized together. Use `-1` for per-channel
  quantization. Defaults to 128.

- num_grid_points:

  Number of grid-search points used to find optimal per-channel scales.
  Higher values can find better scales but take longer. Defaults to 20.

- quantization_layer_structure:

  Optional named list describing the model's quantization structure. It
  should contain `pre_block_layers`, a list of layers to run before the
  first block, and `sequential_blocks`, a list of transformer blocks to
  quantize sequentially. If omitted, the model must provide
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

## References

- [Lin et al., 2023, *AWQ: Activation-aware Weight Quantization for LLM
  Compression and Acceleration*](https://arxiv.org/abs/2306.00978)

- [Reference implementation](https://github.com/mit-han-lab/llm-awq)

## See also

Other quantizers:  
[`quantizer_abs_max_quantize_grouped_with_zero_point()`](https://keras3.posit.co/dev/reference/quantizer_abs_max_quantize_grouped_with_zero_point.md)  
[`quantizer_float8_quantization_config()`](https://keras3.posit.co/dev/reference/quantizer_float8_quantization_config.md)  
[`quantizer_gptq_config()`](https://keras3.posit.co/dev/reference/quantizer_gptq_config.md)  
[`quantizer_int4_quantization_config()`](https://keras3.posit.co/dev/reference/quantizer_int4_quantization_config.md)  
[`quantizer_int8_quantization_config()`](https://keras3.posit.co/dev/reference/quantizer_int8_quantization_config.md)  
[`quantizer_quantization_config()`](https://keras3.posit.co/dev/reference/quantizer_quantization_config.md)  
