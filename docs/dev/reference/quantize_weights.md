# Quantize the weights of a model.

Note that the model must be built first before calling this method.
`quantize_weights()` will recursively call `layer$quantize(...)` in all
layers and will be skipped if the layer doesn't implement the function.

Pass a `mode` string to use the default configuration for that mode.
Advanced users can pass an upstream Keras quantization configuration
object via `config`.

## Usage

``` r
quantize_weights(object, mode = NULL, config = NULL, filters = NULL, ...)
```

## Arguments

- object:

  A Keras Model or Layer.

- mode:

  Quantization mode supported by the installed Keras version. Optional
  when `config` is supplied.

- config:

  An optional upstream Keras quantization configuration object.

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

## See also

Other layer methods:  
[`count_params()`](https://keras3.posit.co/dev/reference/count_params.md)  
[`get_config()`](https://keras3.posit.co/dev/reference/get_config.md)  
[`get_weights()`](https://keras3.posit.co/dev/reference/get_weights.md)  
[`reset_state()`](https://keras3.posit.co/dev/reference/reset_state.md)  
