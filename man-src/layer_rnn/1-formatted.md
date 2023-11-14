Base class for recurrent layers.

@description

# Call Arguments
- `inputs`: Input tensor.
- `initial_state`: List of initial state tensors to be passed to the first
    call of the cell.
- `mask`: Binary tensor of shape `[batch_size, timesteps]`
    indicating whether a given timestep should be masked.
    An individual `True` entry indicates that the corresponding
    timestep should be utilized, while a `False` entry indicates
    that the corresponding timestep should be ignored.
- `training`: Python boolean indicating whether the layer should behave in
    training mode or in inference mode. This argument is passed
    to the cell when calling it.
    This is for use with cells that use dropout.

# Input Shape
3-D tensor with shape `(batch_size, timesteps, features)`.

# Output Shape
- If `return_state`: a list of tensors. The first tensor is
the output. The remaining tensors are the last states,
each with shape `(batch_size, state_size)`, where `state_size` could
be a high dimension tensor shape.
- If `return_sequences`: 3D tensor with shape
`(batch_size, timesteps, output_size)`.

Masking:

This layer supports masking for input data with a variable number
of timesteps. To introduce masks to your data,
use a `keras.layers.Embedding` layer with the `mask_zero` parameter
set to `True`.

Note on using statefulness in RNNs:

You can set RNN layers to be 'stateful', which means that the states
computed for the samples in one batch will be reused as initial states
for the samples in the next batch. This assumes a one-to-one mapping
between samples in different successive batches.

To enable statefulness:

- Specify `stateful=True` in the layer constructor.
- Specify a fixed batch size for your model, by passing
If sequential model:
    `batch_input_shape=(...)` to the first layer in your model.
Else for functional model with 1 or more Input layers:
    `batch_shape=(...)` to all the first layers in your model.
This is the expected shape of your inputs
*including the batch size*.
It should be a tuple of integers, e.g. `(32, 10, 100)`.
- Specify `shuffle=False` when calling `fit()`.

To reset the states of your model, call `.reset_states()` on either
a specific layer, or on your entire model.

Note on specifying the initial state of RNNs:

You can specify the initial state of RNN layers symbolically by
calling them with the keyword argument `initial_state`. The value of
`initial_state` should be a tensor or list of tensors representing
the initial state of the RNN layer.

You can specify the initial state of RNN layers numerically by
calling `reset_states` with the keyword argument `states`. The value of
`states` should be a numpy array or list of numpy arrays representing
the initial state of the RNN layer.

# Examples
```python
from keras.layers import RNN
from keras import ops

# First, let's define a RNN Cell, as a layer subclass.
class MinimalRNNCell(keras.layers.Layer):

    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.state_size = units

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(input_shape[-1], self.units),
                                      initializer='uniform',
                                      name='kernel')
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units),
            initializer='uniform',
            name='recurrent_kernel')
        self.built = True

    def call(self, inputs, states):
        prev_output = states[0]
        h = ops.matmul(inputs, self.kernel)
        output = h + ops.matmul(prev_output, self.recurrent_kernel)
        return output, [output]

# Let's use this cell in a RNN layer:

cell = MinimalRNNCell(32)
x = keras.Input((None, 5))
layer = RNN(cell)
y = layer(x)

# Here's how to use the cell to build a stacked RNN:

cells = [MinimalRNNCell(32), MinimalRNNCell(64)]
x = keras.Input((None, 5))
layer = RNN(cells)
y = layer(x)
```

@param cell
A RNN cell instance or a list of RNN cell instances.
A RNN cell is a class that has:
- A `call(input_at_t, states_at_t)` method, returning
`(output_at_t, states_at_t_plus_1)`. The call method of the
cell can also take the optional argument `constants`, see
section "Note on passing external constants" below.
- A `state_size` attribute. This can be a single integer
(single state) in which case it is the size of the recurrent
state. This can also be a list/tuple of integers
(one size per state).
- A `output_size` attribute, a single integer.
- A `get_initial_state(batch_size=None)`
method that creates a tensor meant to be fed to `call()` as the
initial state, if the user didn't specify any initial state
via other means. The returned initial state should have
shape `(batch_size, cell.state_size)`.
The cell might choose to create a tensor full of zeros,
or other values based on the cell's implementation.
`inputs` is the input tensor to the RNN layer, with shape
`(batch_size, timesteps, features)`.
If this method is not implemented
by the cell, the RNN layer will create a zero filled tensor
with shape `(batch_size, cell.state_size)`.
In the case that `cell` is a list of RNN cell instances, the cells
will be stacked on top of each other in the RNN, resulting in an
efficient stacked RNN.

@param return_sequences
Boolean (default `False`). Whether to return the last
output in the output sequence, or the full sequence.

@param return_state
Boolean (default `False`).
Whether to return the last state in addition to the output.

@param go_backwards
Boolean (default `False`).
If `True`, process the input sequence backwards and return the
reversed sequence.

@param stateful
Boolean (default `False`). If True, the last state
for each sample at index `i` in a batch will be used as initial
state for the sample of index `i` in the following batch.

@param unroll
Boolean (default `False`).
If True, the network will be unrolled, else a symbolic loop will be
used. Unrolling can speed-up a RNN, although it tends to be more
memory-intensive. Unrolling is only suitable for short sequences.

@param zero_output_for_mask
Boolean (default `False`).
Whether the output should use zeros for the masked timesteps.
Note that this field is only used when `return_sequences`
is `True` and `mask` is provided.
It can useful if you want to reuse the raw output sequence of
the RNN without interference from the masked timesteps, e.g.,
merging bidirectional RNNs.

@param object
Object to compose the layer with. A tensor, array, or sequential model.

@param ...
Passed on to the Python callable

@export
@family rnn layers
@family layers
@seealso
+ <https:/keras.io/api/layers/recurrent_layers/rnn#rnn-class>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/layers/RNN>
