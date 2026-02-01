# Reset the state for a model, layer or metric.

Reset the state for a model, layer or metric.

## Usage

``` r
reset_state(object)
```

## Arguments

- object:

  Model, Layer, or Metric instance

  Not all Layers have resettable state (E.g.,
  [`adapt()`](https://keras3.posit.co/reference/adapt.md)-able
  preprocessing layers and rnn layers have resettable state, but a
  [`layer_dense()`](https://keras3.posit.co/reference/layer_dense.md)
  does not). Calling this on a Layer instance without any
  resettable-state will error.

## Value

`object`, invisibly.

## See also

Other layer methods:  
[`count_params()`](https://keras3.posit.co/reference/count_params.md)  
[`get_config()`](https://keras3.posit.co/reference/get_config.md)  
[`get_weights()`](https://keras3.posit.co/reference/get_weights.md)  
[`quantize_weights()`](https://keras3.posit.co/reference/quantize_weights.md)  
