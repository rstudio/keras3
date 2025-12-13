# layer_time_distributed

`time_distributed()` is an alias for
[`layer_time_distributed()`](https://keras3.posit.co/dev/reference/layer_time_distributed.md).
See
`?`[`layer_time_distributed()`](https://keras3.posit.co/dev/reference/layer_time_distributed.md)
for the full documentation.

## Usage

``` r
time_distributed(object, layer, ...)
```

## Arguments

- object:

  Object to compose the layer with. A tensor, array, or sequential
  model.

- layer:

  A `Layer` instance.

- ...:

  For forward/backward compatability.

## Value

The return value depends on the value provided for the first argument.
If `object` is:

- a
  [`keras_model_sequential()`](https://keras3.posit.co/dev/reference/keras_model_sequential.md),
  then the layer is added to the sequential model (which is modified in
  place). To enable piping, the sequential model is also returned,
  invisibly.

- a
  [`keras_input()`](https://keras3.posit.co/dev/reference/keras_input.md),
  then the output tensor from calling `layer(input)` is returned.

- `NULL` or missing, then a `Layer` instance is returned.
