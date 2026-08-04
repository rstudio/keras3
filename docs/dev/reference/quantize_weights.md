# Quantize the weights of a model.

Note that the model must be built first before calling this method.
`quantize_weights()` will recursively call `layer$quantize(...)` in all
layers and will be skipped if the layer doesn't implement the function.

Pass a `mode` string to use the default configuration for int8, int4, or
float8 quantization. AWQ and GPTQ require a corresponding `config`
object. A `config` can also customize the quantizers used for weights
and activations.

## Usage

``` r
quantize_weights(object, mode = NULL, config = NULL, filters = NULL, ...)
```

## Arguments

- object:

  A Keras Model or Layer.

- mode:

  Quantization mode. Supported modes are `"int8"`, `"int4"`, `"float8"`,
  `"gptq"`, and `"awq"`. GPTQ and AWQ require a corresponding `config`.
  Optional when `config` is supplied.

- config:

  A `keras.quantizers.QuantizationConfig` object specifying additional
  quantization options, including custom weight and activation
  quantizers.

- filters:

  Optional filters controlling which layers are quantized. May be a
  regular expression string, a list of regular expression strings, or a
  callable. Only layers matching the filter conditions are quantized.

- ...:

  Passed on to the `object` quantization method.

## Value

`model`, invisibly. Note this is just a convenience for usage with `|>`,
the model is modified in-place.

## Examples

Quantize a model to int8 with the default configuration:

    model <- keras_model_sequential(input_shape = 10) |>
      layer_dense(10)
    model |> quantize_weights("int8")

Quantize with a custom configuration:

    model <- keras_model_sequential(input_shape = 10) |>
      layer_dense(10)
    config <- quantizer_int4_quantization_config(block_size = 64)
    model |> quantize_weights(config = config)

## See also

Other layer methods:  
[`count_params()`](https://keras3.posit.co/dev/reference/count_params.md)  
[`get_config()`](https://keras3.posit.co/dev/reference/get_config.md)  
[`get_weights()`](https://keras3.posit.co/dev/reference/get_weights.md)  
[`reset_state()`](https://keras3.posit.co/dev/reference/reset_state.md)  
