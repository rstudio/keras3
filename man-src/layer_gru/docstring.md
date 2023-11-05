Gated Recurrent Unit - Cho et al. 2014.

Based on available runtime hardware and constraints, this layer
will choose different implementations (cuDNN-based or backend-native)
to maximize the performance. If a GPU is available and all
the arguments to the layer meet the requirement of the cuDNN kernel
(see below for details), the layer will use a fast cuDNN implementation
when using the TensorFlow backend.

The requirements to use the cuDNN implementation are:

1. `activation` == `tanh`
2. `recurrent_activation` == `sigmoid`
3. `dropout` == 0 and `recurrent_dropout` == 0
4. `unroll` is `False`
5. `use_bias` is `True`
6. `reset_after` is `True`
7. Inputs, if use masking, are strictly right-padded.
8. Eager execution is enabled in the outermost context.

There are two variants of the GRU implementation. The default one is based
on [v3](https://arxiv.org/abs/1406.1078v3) and has reset gate applied to
hidden state before matrix multiplication. The other one is based on
[original](https://arxiv.org/abs/1406.1078v1) and has the order reversed.

The second variant is compatible with CuDNNGRU (GPU-only) and allows
inference on CPU. Thus it has separate biases for `kernel` and
`recurrent_kernel`. To use this variant, set `reset_after=True` and
`recurrent_activation='sigmoid'`.

For example:

>>> inputs = np.random.random((32, 10, 8))
>>> gru = keras.layers.GRU(4)
>>> output = gru(inputs)
>>> output.shape
(32, 4)
>>> gru = keras.layers.GRU(4, return_sequences=True, return_state=True)
>>> whole_sequence_output, final_state = gru(inputs)
>>> whole_sequence_output.shape
(32, 10, 4)
>>> final_state.shape
(32, 4)

Args:
    units: Positive integer, dimensionality of the output space.
    activation: Activation function to use.
        Default: hyperbolic tangent (`tanh`).
        If you pass `None`, no activation is applied
        (ie. "linear" activation: `a(x) = x`).
    recurrent_activation: Activation function to use
        for the recurrent step.
        Default: sigmoid (`sigmoid`).
        If you pass `None`, no activation is applied
        (ie. "linear" activation: `a(x) = x`).
    use_bias: Boolean, (default `True`), whether the layer
        should use a bias vector.
    kernel_initializer: Initializer for the `kernel` weights matrix,
        used for the linear transformation of the inputs. Default:
        `"glorot_uniform"`.
    recurrent_initializer: Initializer for the `recurrent_kernel`
        weights matrix, used for the linear transformation of the recurrent
        state. Default: `"orthogonal"`.
    bias_initializer: Initializer for the bias vector. Default: `"zeros"`.
    kernel_regularizer: Regularizer function applied to the `kernel` weights
        matrix. Default: `None`.
    recurrent_regularizer: Regularizer function applied to the
        `recurrent_kernel` weights matrix. Default: `None`.
    bias_regularizer: Regularizer function applied to the bias vector.
        Default: `None`.
    activity_regularizer: Regularizer function applied to the output of the
        layer (its "activation"). Default: `None`.
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
    seed: Random seed for dropout.
    return_sequences: Boolean. Whether to return the last output
        in the output sequence, or the full sequence. Default: `False`.
    return_state: Boolean. Whether to return the last state in addition
        to the output. Default: `False`.
    go_backwards: Boolean (default `False`).
        If `True`, process the input sequence backwards and return the
        reversed sequence.
    stateful: Boolean (default: `False`). If `True`, the last state
        for each sample at index i in a batch will be used as initial
        state for the sample of index i in the following batch.
    unroll: Boolean (default: `False`).
        If `True`, the network will be unrolled,
        else a symbolic loop will be used.
        Unrolling can speed-up a RNN,
        although it tends to be more memory-intensive.
        Unrolling is only suitable for short sequences.
    reset_after: GRU convention (whether to apply reset gate after or
        before matrix multiplication). `False` is `"before"`,
        `True` is `"after"` (default and cuDNN compatible).

Call arguments:
    inputs: A 3D tensor, with shape `(batch, timesteps, feature)`.
    mask: Binary tensor of shape `(samples, timesteps)` indicating whether
        a given timestep should be masked  (optional).
        An individual `True` entry indicates that the corresponding timestep
        should be utilized, while a `False` entry indicates that the
        corresponding timestep should be ignored. Defaults to `None`.
    training: Python boolean indicating whether the layer should behave in
        training mode or in inference mode. This argument is passed to the
        cell when calling it. This is only relevant if `dropout` or
        `recurrent_dropout` is used  (optional). Defaults to `None`.
    initial_state: List of initial state tensors to be passed to the first
        call of the cell (optional, `None` causes creation
        of zero-filled initial state tensors). Defaults to `None`.
