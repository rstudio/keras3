# layer_bidirectional

`bidirectional()` is an alias for
[`layer_bidirectional()`](https://keras3.posit.co/dev/reference/layer_bidirectional.md).
See
`?`[`layer_bidirectional()`](https://keras3.posit.co/dev/reference/layer_bidirectional.md)
for the full documentation.

## Usage

``` r
bidirectional(
  object,
  layer,
  merge_mode = "concat",
  weights = NULL,
  backward_layer = NULL,
  ...
)
```

## Arguments

- object:

  Object to compose the layer with. A tensor, array, or sequential
  model.

- layer:

  `RNN` instance, such as
  [`layer_lstm()`](https://keras3.posit.co/dev/reference/layer_lstm.md)
  or
  [`layer_gru()`](https://keras3.posit.co/dev/reference/layer_gru.md).
  It could also be a
  [`Layer()`](https://keras3.posit.co/dev/reference/Layer.md) instance
  that meets the following criteria:

  1.  Be a sequence-processing layer (accepts 3D+ inputs).

  2.  Have a `go_backwards`, `return_sequences` and `return_state`
      attribute (with the same semantics as for the `RNN` class).

  3.  Have an `input_spec` attribute.

  4.  Implement serialization via
      [`get_config()`](https://keras3.posit.co/dev/reference/get_config.md)
      and
      [`from_config()`](https://keras3.posit.co/dev/reference/get_config.md).
      Note that the recommended way to create new RNN layers is to write
      a custom RNN cell and use it with
      [`layer_rnn()`](https://keras3.posit.co/dev/reference/layer_rnn.md),
      instead of subclassing with
      [`Layer()`](https://keras3.posit.co/dev/reference/Layer.md)
      directly. When `return_sequences` is `TRUE`, the output of the
      masked timestep will be zero regardless of the layer's original
      `zero_output_for_mask` value.

- merge_mode:

  Mode by which outputs of the forward and backward RNNs will be
  combined. One of `{"sum", "mul", "concat", "ave", NULL}`. If `NULL`,
  the outputs will not be combined, they will be returned as a list.
  Defaults to `"concat"`.

- weights:

  see description

- backward_layer:

  Optional `RNN`, or
  [`Layer()`](https://keras3.posit.co/dev/reference/Layer.md) instance
  to be used to handle backwards input processing. If `backward_layer`
  is not provided, the layer instance passed as the `layer` argument
  will be used to generate the backward layer automatically. Note that
  the provided `backward_layer` layer should have properties matching
  those of the `layer` argument, in particular it should have the same
  values for `stateful`, `return_states`, `return_sequences`, etc. In
  addition, `backward_layer` and `layer` should have different
  `go_backwards` argument values. A `ValueError` will be raised if these
  requirements are not met.

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
