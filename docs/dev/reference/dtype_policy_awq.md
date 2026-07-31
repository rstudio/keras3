# Quantized dtype policies

These policies propagate quantization settings when loading quantized
models in Keras format.

## Usage

``` r
dtype_policy_awq(mode, source_name = NULL)

dtype_policy_gptq(mode, source_name = NULL)

dtype_policy_int4(mode, source_name = NULL)
```

## Arguments

- mode:

  Quantization mode. For AWQ and GPTQ, use
  `"<algorithm>/<weight_bits>/<group_size>"`, such as `"awq/4/128"`. For
  int4, use `"int4/<block_size>"`, such as `"int4/128"`.

- source_name:

  Optional source dtype policy name, such as `"float32"`.

## Value

A quantized `DTypePolicy` instance.

## Examples

    awq_policy <- dtype_policy_awq("awq/4/128")
    gptq_policy <- dtype_policy_gptq("gptq/4/128")
    int4_policy <- dtype_policy_int4("int4/128")
