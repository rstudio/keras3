keras.layers.Bidirectional
__signature__
(
  layer,
  merge_mode='concat',
  weights=None,
  backward_layer=None,
  **kwargs
)
__doc__
Bidirectional wrapper for RNNs.

Args:
    layer: `keras.layers.RNN` instance, such as
        `keras.layers.LSTM` or `keras.layers.GRU`.
        It could also be a `keras.layers.Layer` instance
        that meets the following criteria:
        1. Be a sequence-processing layer (accepts 3D+ inputs).
        2. Have a `go_backwards`, `return_sequences` and `return_state`
        attribute (with the same semantics as for the `RNN` class).
        3. Have an `input_spec` attribute.
        4. Implement serialization via `get_config()` and `from_config()`.
        Note that the recommended way to create new RNN layers is to write a
        custom RNN cell and use it with `keras.layers.RNN`, instead of
        subclassing `keras.layers.Layer` directly.
        When `return_sequences` is `True`, the output of the masked
        timestep will be zero regardless of the layer's original
        `zero_output_for_mask` value.
    merge_mode: Mode by which outputs of the forward and backward RNNs
        will be combined. One of `{"sum", "mul", "concat", "ave", None}`.
        If `None`, the outputs will not be combined,
        they will be returned as a list. Defaults to `"concat"`.
    backward_layer: Optional `keras.layers.RNN`,
        or `keras.layers.Layer` instance to be used to handle
        backwards input processing.
        If `backward_layer` is not provided, the layer instance passed
        as the `layer` argument will be used to generate the backward layer
        automatically.
        Note that the provided `backward_layer` layer should have properties
        matching those of the `layer` argument, in particular
        it should have the same values for `stateful`, `return_states`,
        `return_sequences`, etc. In addition, `backward_layer`
        and `layer` should have different `go_backwards` argument values.
        A `ValueError` will be raised if these requirements are not met.

Call arguments:
    The call arguments for this layer are the same as those of the
    wrapped RNN layer. Beware that when passing the `initial_state`
    argument during the call of this layer, the first half in the
    list of elements in the `initial_state` list will be passed to
    the forward RNN call and the last half in the list of elements
    will be passed to the backward RNN call.

Note: instantiating a `Bidirectional` layer from an existing RNN layer
instance will not reuse the weights state of the RNN layer instance -- the
`Bidirectional` layer will have freshly initialized weights.

Examples:

```python
model = Sequential([
    Input(shape=(5, 10)),
    Bidirectional(LSTM(10, return_sequences=True),
    Bidirectional(LSTM(10)),
    Dense(5, activation="softmax"),
])
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

# With custom backward layer
forward_layer = LSTM(10, return_sequences=True)
backward_layer = LSTM(10, activation='relu', return_sequences=True,
                      go_backwards=True)
model = Sequential([
    Input(shape=(5, 10)),
    Bidirectional(forward_layer, backward_layer=backward_layer),
    Dense(5, activation="softmax"),
])
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
```
