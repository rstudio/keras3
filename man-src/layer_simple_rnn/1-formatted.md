Fully-connected RNN where the output is to be fed back as the new input.

@description

# Call Arguments
- `sequence`: A 3D tensor, with shape `[batch, timesteps, feature]`.
- `mask`: Binary tensor of shape `[batch, timesteps]` indicating whether
    a given timestep should be masked. An individual `True` entry
    indicates that the corresponding timestep should be utilized,
    while a `False` entry indicates that the corresponding timestep
    should be ignored.
- `training`: Python boolean indicating whether the layer should behave in
    training mode or in inference mode.
    This argument is passed to the cell when calling it.
    This is only relevant if `dropout` or `recurrent_dropout` is used.
- `initial_state`: List of initial state tensors to be passed to the first
    call of the cell.

# Examples
```python
inputs = np.random.random((32, 10, 8))
simple_rnn = keras.layers.SimpleRNN(4)
output = simple_rnn(inputs)  # The output has shape `(32, 4)`.
simple_rnn = keras.layers.SimpleRNN(
    4, return_sequences=True, return_state=True
)
# whole_sequence_output has shape `(32, 10, 4)`.
# final_state has shape `(32, 4)`.
whole_sequence_output, final_state = simple_rnn(inputs)
```

@param units
Positive integer, dimensionality of the output space.

@param activation
Activation function to use.
Default: hyperbolic tangent (`tanh`).
If you pass None, no activation is applied
(ie. "linear" activation: `a(x) = x`).

@param use_bias
Boolean, (default `True`), whether the layer uses
a bias vector.

@param kernel_initializer
Initializer for the `kernel` weights matrix,
used for the linear transformation of the inputs. Default:
`"glorot_uniform"`.

@param recurrent_initializer
Initializer for the `recurrent_kernel`
weights matrix, used for the linear transformation of the recurrent
state.  Default: `"orthogonal"`.

@param bias_initializer
Initializer for the bias vector. Default: `"zeros"`.

@param kernel_regularizer
Regularizer function applied to the `kernel` weights
matrix. Default: `None`.

@param recurrent_regularizer
Regularizer function applied to the
`recurrent_kernel` weights matrix. Default: `None`.

@param bias_regularizer
Regularizer function applied to the bias vector.
Default: `None`.

@param activity_regularizer
Regularizer function applied to the output of the
layer (its "activation"). Default: `None`.

@param kernel_constraint
Constraint function applied to the `kernel` weights
matrix. Default: `None`.

@param recurrent_constraint
Constraint function applied to the
`recurrent_kernel` weights matrix.  Default: `None`.

@param bias_constraint
Constraint function applied to the bias vector.
Default: `None`.

@param dropout
Float between 0 and 1.
Fraction of the units to drop for the linear transformation
of the inputs. Default: 0.

@param recurrent_dropout
Float between 0 and 1.
Fraction of the units to drop for the linear transformation of the
recurrent state. Default: 0.

@param return_sequences
Boolean. Whether to return the last output
in the output sequence, or the full sequence. Default: `False`.

@param return_state
Boolean. Whether to return the last state
in addition to the output. Default: `False`.

@param go_backwards
Boolean (default: `False`).
If `True`, process the input sequence backwards and return the
reversed sequence.

@param stateful
Boolean (default: `False`). If `True`, the last state
for each sample at index i in a batch will be used as initial
state for the sample of index i in the following batch.

@param unroll
Boolean (default: `False`).
If `True`, the network will be unrolled,
else a symbolic loop will be used.
Unrolling can speed-up a RNN,
although it tends to be more memory-intensive.
Unrolling is only suitable for short sequences.

@param object
Object to compose the layer with. A tensor, array, or sequential model.

@param seed
Initial seed for the random number generator

@param ...
Passed on to the Python callable

@export
@family rnn layers
@family layers
@seealso
+ <https:/keras.io/api/layers/recurrent_layers/simple_rnn#simplernn-class>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/layers/SimpleRNN>
