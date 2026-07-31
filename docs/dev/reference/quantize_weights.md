# Quantize the weights of a model.

Note that the model must be built first before calling this method.
`quantize_weights()` will recursively call `layer$quantize(...)` in all
layers and will be skipped if the layer doesn't implement the function.

Available quantization modes and supported layer types depend on the
current Keras backend and release.

## Usage

``` r
quantize_weights(object, mode = NULL, config = NULL, filters = NULL, ...)
```

## Arguments

- object:

  A Keras Model or Layer.

- mode:

  Optional quantization mode. Recent Keras releases support modes such
  as `"int8"`, `"int4"`, `"float8"`, and `"gptq"`.

- config:

  Optional quantization configuration object. When supplied, this can be
  used instead of `mode` to customize the quantizers used for weights or
  activations.

- filters:

  Optional filters controlling which layers are quantized. This may be a
  regex string, a list of regex strings, or a callable.

- ...:

  Passed on to the `object` quantization method.

## Value

`model`, invisibly. Note this is just a convenience for usage with `|>`,
the model is modified in-place.

## See also

Other layer methods:  
[`count_params()`](https://keras3.posit.co/dev/reference/count_params.md)  
[`get_config()`](https://keras3.posit.co/dev/reference/get_config.md)  
[`get_weights()`](https://keras3.posit.co/dev/reference/get_weights.md)  
[`reset_state()`](https://keras3.posit.co/dev/reference/reset_state.md)  
