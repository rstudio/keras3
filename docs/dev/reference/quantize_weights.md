# Quantize the weights of a model.

Note that the model must be built first before calling this method.
`quantize_weights()` will recursively call `layer$quantize(mode)` in all
layers and will be skipped if the layer doesn't implement the function.

Currently only `Dense` and `EinsumDense` layers support quantization.

## Usage

``` r
quantize_weights(object, mode, ...)
```

## Arguments

- object:

  A Keras Model or Layer.

- mode:

  The mode of the quantization. Only 'int8' is supported at this time.

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
