#' Quantized dtype policies
#'
#' These policies propagate AWQ, GPTQ, or int4 quantization settings when
#' loading a quantized model in Keras format.
#'
#' # Examples
#'
#' ```{r}
#' awq_policy <- dtype_policy_awq("awq/4/128")
#' gptq_policy <- dtype_policy_gptq("gptq/4/128")
#' int4_policy <- dtype_policy_int4("int4/128")
#' ```
#'
#' @param mode Quantization mode, with a format determined by the policy:
#'
#'   - For [`dtype_policy_awq()`], use
#'     `"awq/<weight_bits>/<group_size>"`. AWQ supports 4-bit weights.
#'     A `<group_size>` of `-1` selects per-channel quantization; any positive
#'     integer selects grouped quantization. For example, `"awq/4/128"`.
#'   - For [`dtype_policy_gptq()`], use
#'     `"gptq/<weight_bits>/<group_size>"`. GPTQ supports 2-, 3-, 4-, and 8-bit
#'     weights. A `<group_size>` of `-1` selects whole-tensor quantization; any
#'     positive integer selects grouped quantization. Smaller groups typically
#'     give better accuracy but slower speed. For example, `"gptq/4/128"`.
#'   - For [`dtype_policy_int4()`], use `"int4/<block_size>"`.
#'     A `<block_size>` of `-1` selects legacy per-channel quantization; any
#'     positive integer selects sub-channel quantization with that block size.
#'     For example, `"int4/128"` uses 128-element groups.
#' @param source_name Optional source dtype policy name, such as `"float32"`.
#'
#' @returns A quantized `DTypePolicy` instance.
#' @export
#' @family dtype policies
#' @tether keras.dtype_policies.AWQDTypePolicy
#' @name dtype_policy_awq
dtype_policy_awq <-
function(mode, source_name = NULL)
{
  args <- capture_args()
  do.call(keras$dtype_policies$AWQDTypePolicy, args)
}


#' @rdname dtype_policy_awq
#' @export
#' @tether keras.dtype_policies.GPTQDTypePolicy
dtype_policy_gptq <-
function(mode, source_name = NULL)
{
  args <- capture_args()
  do.call(keras$dtype_policies$GPTQDTypePolicy, args)
}


#' @rdname dtype_policy_awq
#' @export
#' @tether keras.dtype_policies.Int4DTypePolicy
dtype_policy_int4 <-
function(mode, source_name = NULL)
{
  args <- capture_args()
  do.call(keras$dtype_policies$Int4DTypePolicy, args)
}
