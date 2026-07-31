#' Base configuration for model quantization
#'
#' Subclasses provide a quantization `mode` and serialization methods. Use a
#' concrete configuration such as [`quantizer_int8_quantization_config()`] for
#' model quantization. This base class does not define a quantization mode.
#'
#' @param weight_quantizer Optional quantizer for weights.
#' @param activation_quantizer Optional quantizer for activations.
#'
#' @returns A `QuantizationConfig` instance.
#' @export
#' @family quantizers
#' @tether keras.quantizers.QuantizationConfig
quantizer_quantization_config <-
function(weight_quantizer = NULL, activation_quantizer = NULL)
{
  args <- capture_args()
  do.call(keras$quantizers$QuantizationConfig, args)
}


#' Float8 quantization configuration
#'
#' Float8 mixed-precision training does not support user-defined quantizers;
#' this object selects the built-in float8 behavior.
#'
#' # Examples
#'
#' ```{r}
#' config <- quantizer_float8_quantization_config()
#' ```
#'
#' @returns A `Float8QuantizationConfig` instance.
#' @export
#' @family quantizers
#' @tether keras.quantizers.Float8QuantizationConfig
quantizer_float8_quantization_config <-
function()
{
  args <- capture_args()
  do.call(keras$quantizers$Float8QuantizationConfig, args)
}


#' Int4 quantization configuration
#'
#' Configures weight-only int4 quantization by default, using groups along the
#' input dimension.
#'
#' # Examples
#'
#' ```{r}
#' config <- quantizer_int4_quantization_config(block_size = 128)
#' ```
#'
#' @param weight_quantizer Optional quantizer for weights.
#' @param activation_quantizer Optional quantizer for activations. The default
#'   selects weight-only int4 quantization.
#' @param block_size Group size along the input dimension. A positive integer
#'   uses sub-channel quantization with
#'   `ceiling(input_dim / block_size)` groups; `NULL` or `-1` uses per-channel
#'   quantization with one scale per output channel. Sub-channel quantization
#'   does not support custom weight or activation quantizers. Defaults to 128.
#'
#' @returns An `Int4QuantizationConfig` instance.
#' @export
#' @family quantizers
#' @tether keras.quantizers.Int4QuantizationConfig
quantizer_int4_quantization_config <-
function(weight_quantizer = NULL, activation_quantizer = "default",
         block_size = 128L)
{
  args <- capture_args(list(block_size = as_integer))
  do.call(keras$quantizers$Int4QuantizationConfig, args)
}


#' Int8 quantization configuration
#'
#' Configures int8 quantization with an absolute-maximum activation quantizer
#' by default.
#'
#' # Examples
#'
#' ```{r}
#' config <- quantizer_int8_quantization_config()
#' ```
#'
#' @param weight_quantizer Optional quantizer for weights.
#' @param activation_quantizer Optional quantizer for activations. The default
#'   uses an absolute-maximum quantizer over the last axis.
#'
#' @returns An `Int8QuantizationConfig` instance.
#' @export
#' @family quantizers
#' @tether keras.quantizers.Int8QuantizationConfig
quantizer_int8_quantization_config <-
function(weight_quantizer = NULL, activation_quantizer = "default")
{
  args <- capture_args()
  do.call(keras$quantizers$Int8QuantizationConfig, args)
}


#' AWQ calibration and quantization configuration
#'
#' Activation-aware Weight Quantization is a post-training method that
#' identifies and protects salient weights based on activation magnitudes. It
#' uses calibration data to collect activation statistics, identifies salient
#' weight channels, searches a grid for optimal per-channel scales, and applies
#' those scales before quantization to reduce accuracy loss.
#'
#' # Examples
#'
#' ```{r, eval = FALSE}
#' config <- quantizer_awq_config(
#'   dataset = calibration_text,
#'   tokenizer = tokenizer,
#'   group_size = 128
#' )
#' model |> quantize_weights(mode = "awq", config = config)
#' ```
#'
#' @param dataset Calibration data used to analyze activation patterns. It can
#'   be an iterable that yields strings or pre-tokenized numeric tensors, such
#'   as a character vector, generator, R array, or NumPy array.
#' @param tokenizer Tokenizer or compatible callable used to process `dataset`
#'   when it contains strings.
#' @param ... For forward/backward compatibility. Arguments after `...` must be
#'   named.
#' @param weight_bits Number of bits used for weight quantization. AWQ currently
#'   supports only 4. Defaults to 4.
#' @param num_samples Number of calibration samples to use. Defaults to 128.
#' @param sequence_length Sequence length of each calibration sample. Defaults
#'   to 512.
#' @param group_size Number of weights quantized together. Use `-1` for
#'   per-channel quantization. Defaults to 128.
#' @param num_grid_points Number of grid-search points used to find optimal
#'   per-channel scales. Higher values can find better scales but take longer.
#'   Defaults to 20.
#' @param quantization_layer_structure Optional named list describing
#'   the model's quantization structure. It should contain
#'   `pre_block_layers`, a list of layers to run before the first block, and
#'   `sequential_blocks`, a list of transformer blocks to quantize
#'   sequentially. If omitted, the model must provide
#'   `get_quantization_layer_structure()`.
#'
#' @returns An `AWQConfig` instance.
#' @export
#' @family quantizers
#' @references
#' - [Lin et al., 2023, *AWQ: Activation-aware Weight Quantization for LLM
#'   Compression and Acceleration*](https://arxiv.org/abs/2306.00978)
#' - [Reference implementation](https://github.com/mit-han-lab/llm-awq)
#' @tether keras.quantizers.AWQConfig
quantizer_awq_config <-
function(dataset, tokenizer, ..., weight_bits = 4L, num_samples = 128L,
         sequence_length = 512L, group_size = 128L, num_grid_points = 20L,
         quantization_layer_structure = NULL)
{
  args <- capture_args(list(
    weight_bits = as_integer,
    num_samples = as_integer,
    sequence_length = as_integer,
    group_size = as_integer,
    num_grid_points = as_integer
  ))
  do.call(keras$quantizers$AWQConfig, args)
}


#' GPTQ calibration and quantization configuration
#'
#' GPTQ is a post-training method that quantizes weights to lower precision
#' while reducing the impact on model accuracy. It uses calibration data to
#' estimate a Hessian for each layer, applies iterative quantization with error
#' correction, optionally reorders weights by activation importance, and
#' minimizes quantization error. It can reduce model size and memory use and
#' does not require retraining a pretrained model.
#'
#' Quantization quality depends heavily on the calibration dataset. For best
#' results, use representative data covering the expected input distribution.
#'
#' # Examples
#'
#' ```{r, eval = FALSE}
#' config <- quantizer_gptq_config(
#'   dataset = calibration_text,
#'   tokenizer = tokenizer,
#'   weight_bits = 4,
#'   group_size = 128
#' )
#' model |> quantize_weights(mode = "gptq", config = config)
#' ```
#'
#' @inheritParams quantizer_awq_config
#' @param weight_bits Number of bits used for weight quantization. GPTQ supports
#'   2, 3, 4, and 8. Defaults to 4.
#' @param per_channel Whether to calculate quantization parameters per output
#'   channel. Defaults to `TRUE`.
#' @param hessian_damping Fraction of Hessian damping used to stabilize the
#'   inverse calculation. Must be between 0 and 1. Defaults to 0.01.
#' @param symmetric Whether to use symmetric rather than asymmetric
#'   quantization. Defaults to `FALSE`.
#' @param activation_order Whether to reorder weight columns by activation
#'   magnitude, which can improve quantization accuracy. Defaults to `FALSE`.
#'
#' @returns A `GPTQConfig` instance.
#' @export
#' @family quantizers
#' @references
#' - [Frantar et al., 2022, *GPTQ: Accurate Post-Training Quantization for
#'   Generative Pre-trained Transformers*](https://arxiv.org/abs/2210.17323)
#' - [Reference implementation](https://github.com/IST-DASLab/gptq)
#' @tether keras.quantizers.GPTQConfig
quantizer_gptq_config <-
function(dataset, tokenizer, ..., weight_bits = 4L, num_samples = 128L,
         per_channel = TRUE, sequence_length = 512L,
         hessian_damping = 0.01, group_size = 128L, symmetric = FALSE,
         activation_order = FALSE, quantization_layer_structure = NULL)
{
  args <- capture_args(list(
    weight_bits = as_integer,
    num_samples = as_integer,
    sequence_length = as_integer,
    group_size = as_integer
  ))
  do.call(keras$quantizers$GPTQConfig, args)
}


#' Grouped asymmetric absolute-maximum quantization
#'
#' Quantizes a 2D tensor in row groups. Each group receives a scale and zero
#' point for every output column. Groups are formed along the first axis, the
#' input or contracting dimension. Asymmetric quantization is useful for weight
#' distributions that are not centered around zero.
#'
#' # Examples
#'
#' ```{r}
#' kernel <- op_reshape(op_arange(16, dtype = "float32"), c(4, 4))
#' result <- quantizer_abs_max_quantize_grouped_with_zero_point(
#'   kernel,
#'   block_size = 2
#' )
#' ```
#'
#' @param inputs A 2D tensor with shape `(input_dim, output_dim)`.
#' @param block_size Number of rows in each quantization group.
#' @param value_range Integer vector giving the minimum and maximum quantized
#'   values.
#' @param dtype Dtype of the quantized output.
#' @param epsilon Small value used to avoid division by zero.
#' @param to_numpy Whether to perform the computation in NumPy for memory
#'   efficiency.
#'
#' @returns A list containing:
#'
#'   - `quantized_tensor`: a tensor with the same shape as `inputs` and dtype
#'     `dtype`.
#'   - `scale`: a tensor with shape `(n_groups, output_dim)`, where
#'     `n_groups = ceiling(input_dim / block_size)`.
#'   - `zero_point`: a `uint8` tensor with shape
#'     `(n_groups, output_dim)`.
#' @export
#' @family quantizers
#' @tether keras.quantizers.abs_max_quantize_grouped_with_zero_point
quantizer_abs_max_quantize_grouped_with_zero_point <-
function(inputs, block_size, value_range = c(-8L, 7L), dtype = "int8",
         epsilon = 1e-7, to_numpy = FALSE)
{
  args <- capture_args(list(
    block_size = as_integer,
    value_range = as_integer_tuple
  ))
  do.call(
    keras$quantizers$abs_max_quantize_grouped_with_zero_point,
    args
  )
}
