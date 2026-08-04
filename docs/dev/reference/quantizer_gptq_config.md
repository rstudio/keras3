# GPTQ calibration and quantization configuration

GPTQ is a post-training method that quantizes weights to lower precision
while reducing the impact on model accuracy. It uses calibration data to
estimate a Hessian for each layer, applies iterative quantization with
error correction, optionally reorders weights by activation importance,
and minimizes quantization error. It can reduce model size and memory
use and does not require retraining a pretrained model.

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

  Number of bits used for weight quantization. GPTQ supports 2, 3, 4,
  and 8. Defaults to 4.

- num_samples:

  Number of calibration samples to use. Defaults to 128.

- per_channel:

  Whether to calculate quantization parameters per output channel.
  Defaults to `TRUE`.

- sequence_length:

  Sequence length of each calibration sample. Defaults to 512.

- hessian_damping:

  Fraction of Hessian damping used to stabilize the inverse calculation.
  Must be between 0 and 1. Defaults to 0.01.

- group_size:

  Number of weights quantized together. Use `-1` for per-channel
  quantization. Defaults to 128.

- symmetric:

  Whether to use symmetric rather than asymmetric quantization. Defaults
  to `FALSE`.

- activation_order:

  Whether to reorder weight columns by activation magnitude, which can
  improve quantization accuracy. Defaults to `FALSE`.

- quantization_layer_structure:

  Optional named list describing the model's quantization structure. It
  should contain `pre_block_layers`, a list of layers to run before the
  first block, and `sequential_blocks`, a list of transformer blocks to
  quantize sequentially. If omitted, the model must provide
  `get_quantization_layer_structure()`.

## Value

A `GPTQConfig` instance.

## Details

Quantization quality depends heavily on the calibration dataset. For
best results, use representative data covering the expected input
distribution.

## Examples

    config <- quantizer_gptq_config(
      dataset = calibration_text,
      tokenizer = tokenizer,
      weight_bits = 4,
      group_size = 128
    )
    model |> quantize_weights(mode = "gptq", config = config)

## References

- [Frantar et al., 2022, *GPTQ: Accurate Post-Training Quantization for
  Generative Pre-trained
  Transformers*](https://arxiv.org/abs/2210.17323)

- [Reference implementation](https://github.com/IST-DASLab/gptq)

## See also

Other quantizers:  
[`quantizer_abs_max_quantize_grouped_with_zero_point()`](https://keras3.posit.co/dev/reference/quantizer_abs_max_quantize_grouped_with_zero_point.md)  
[`quantizer_awq_config()`](https://keras3.posit.co/dev/reference/quantizer_awq_config.md)  
[`quantizer_float8_quantization_config()`](https://keras3.posit.co/dev/reference/quantizer_float8_quantization_config.md)  
[`quantizer_int4_quantization_config()`](https://keras3.posit.co/dev/reference/quantizer_int4_quantization_config.md)  
[`quantizer_int8_quantization_config()`](https://keras3.posit.co/dev/reference/quantizer_int8_quantization_config.md)  
[`quantizer_quantization_config()`](https://keras3.posit.co/dev/reference/quantizer_quantization_config.md)  
