Cell class for SimpleRNN.

@description
This class processes one step within the whole time sequence input, whereas
`keras.layer.SimpleRNN` processes the whole sequence.

# Call Arguments
- `sequence`: A 2D tensor, with shape `(batch, features)`.
- `states`: A 2D tensor with shape `(batch, units)`, which is the state
    from the previous time step.
- `training`: Python boolean indicating whether the layer should behave in
    training mode or in inference mode. Only relevant when `dropout` or
    `recurrent_dropout` is used.

# Examples
```python
inputs = np.random.random([32, 10, 8]).astype(np.float32)
rnn = keras.layers.RNN(keras.layers.SimpleRNNCell(4))
output = rnn(inputs)  # The output has shape `(32, 4)`.
rnn = keras.layers.RNN(
    keras.layers.SimpleRNNCell(4),
    return_sequences=True,
    return_state=True
)
# whole_sequence_output has shape `(32, 10, 4)`.
# final_state has shape `(32, 4)`.
whole_sequence_output, final_state = rnn(inputs)
```

@param units
Positive integer, dimensionality of the output space.

@param activation
Activation function to use.
Default: hyperbolic tangent (`tanh`).
If you pass `None`, no activation is applied
(ie. "linear" activation: `a(x) = x`).

@param use_bias
Boolean, (default `True`), whether the layer
should use a bias vector.

@param kernel_initializer
Initializer for the `kernel` weights matrix,
used for the linear transformation of the inputs. Default:
`"glorot_uniform"`.

@param recurrent_initializer
Initializer for the `recurrent_kernel`
weights matrix, used for the linear transformation
of the recurrent state. Default: `"orthogonal"`.

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

@param kernel_constraint
Constraint function applied to the `kernel` weights
matrix. Default: `None`.

@param recurrent_constraint
Constraint function applied to the
`recurrent_kernel` weights matrix. Default: `None`.

@param bias_constraint
Constraint function applied to the bias vector.
Default: `None`.

@param dropout
Float between 0 and 1. Fraction of the units to drop for the
linear transformation of the inputs. Default: 0.

@param recurrent_dropout
Float between 0 and 1. Fraction of the units to drop
for the linear transformation of the recurrent state. Default: 0.

@param seed
Random seed for dropout.

@param ...
Passed on to the Python callable

@export
@family simple rnn layers
@family rnn layers
@family layers
@seealso
+ <https://www.tensorflow.org/api_docs/python/tf/keras/layers/SimpleRNNCell>
