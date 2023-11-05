Cell class for the GRU layer.

This class processes one step within the whole time sequence input, whereas
`keras.layer.GRU` processes the whole sequence.

Args:
    units: Positive integer, dimensionality of the output space.
    activation: Activation function to use. Default: hyperbolic tangent
        (`tanh`). If you pass None, no activation is applied
        (ie. "linear" activation: `a(x) = x`).
    recurrent_activation: Activation function to use for the recurrent step.
        Default: sigmoid (`sigmoid`). If you pass `None`, no activation is
        applied (ie. "linear" activation: `a(x) = x`).
    use_bias: Boolean, (default `True`), whether the layer
        should use a bias vector.
    kernel_initializer: Initializer for the `kernel` weights matrix,
        used for the linear transformation of the inputs. Default:
        `"glorot_uniform"`.
    recurrent_initializer: Initializer for the `recurrent_kernel`
        weights matrix, used for the linear transformation
        of the recurrent state. Default: `"orthogonal"`.
    bias_initializer: Initializer for the bias vector. Default: `"zeros"`.
    kernel_regularizer: Regularizer function applied to the `kernel` weights
        matrix. Default: `None`.
    recurrent_regularizer: Regularizer function applied to the
        `recurrent_kernel` weights matrix. Default: `None`.
    bias_regularizer: Regularizer function applied to the bias vector.
        Default: `None`.
    kernel_constraint: Constraint function applied to the `kernel` weights
        matrix. Default: `None`.
    recurrent_constraint: Constraint function applied to the
        `recurrent_kernel` weights matrix. Default: `None`.
    bias_constraint: Constraint function applied to the bias vector.
        Default: `None`.
    dropout: Float between 0 and 1. Fraction of the units to drop for the
        linear transformation of the inputs. Default: 0.
    recurrent_dropout: Float between 0 and 1. Fraction of the units to drop
        for the linear transformation of the recurrent state. Default: 0.
    reset_after: GRU convention (whether to apply reset gate after or
        before matrix multiplication). False = "before",
        True = "after" (default and cuDNN compatible).
    seed: Random seed for dropout.

Call arguments:
    inputs: A 2D tensor, with shape `(batch, features)`.
    states: A 2D tensor with shape `(batch, units)`, which is the state
        from the previous time step.
    training: Python boolean indicating whether the layer should behave in
        training mode or in inference mode. Only relevant when `dropout` or
        `recurrent_dropout` is used.

Example:

>>> inputs = np.random.random((32, 10, 8))
>>> rnn = keras.layers.RNN(keras.layers.GRUCell(4))
>>> output = rnn(inputs)
>>> output.shape
(32, 4)
>>> rnn = keras.layers.RNN(
...    keras.layers.GRUCell(4),
...    return_sequences=True,
...    return_state=True)
>>> whole_sequence_output, final_state = rnn(inputs)
>>> whole_sequence_output.shape
(32, 10, 4)
>>> final_state.shape
(32, 4)
