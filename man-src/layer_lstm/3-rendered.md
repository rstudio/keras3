Long Short-Term Memory layer - Hochreiter 1997.

@description
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
4. `unroll` is `FALSE`
5. `use_bias` is `TRUE`
6. Inputs, if use masking, are strictly right-padded.
7. Eager execution is enabled in the outermost context.

For example:


```r
input <- random_uniform(c(32, 10, 8))
output <- input |> layer_lstm(4)
output$shape
```

```
## TensorShape([32, 4])
```

```r
lstm <- layer_lstm(units = 4, return_sequences = TRUE, return_state = TRUE)
c(whole_seq_output, final_memory_state, final_carry_state) %<-% lstm(input)
whole_seq_output$shape
```

```
## TensorShape([32, 10, 4])
```

```r
final_memory_state$shape
```

```
## TensorShape([32, 4])
```

```r
final_carry_state$shape
```

```
## TensorShape([32, 4])
```

# Call Arguments
- `inputs`: A 3D tensor, with shape `(batch, timesteps, feature)`.
- `mask`: Binary tensor of shape `(samples, timesteps)` indicating whether
    a given timestep should be masked  (optional).
    An individual `TRUE` entry indicates that the corresponding timestep
    should be utilized, while a `FALSE` entry indicates that the
    corresponding timestep should be ignored. Defaults to `NULL`.
- `training`: Boolean indicating whether the layer should behave in
    training mode or in inference mode. This argument is passed to the
    cell when calling it. This is only relevant if `dropout` or
    `recurrent_dropout` is used  (optional). Defaults to `NULL`.
- `initial_state`: List of initial state tensors to be passed to the first
    call of the cell (optional, `NULL` causes creation
    of zero-filled initial state tensors). Defaults to `NULL`.

@param units Positive integer, dimensionality of the output space.
@param activation Activation function to use.
    Default: hyperbolic tangent (`tanh`).
    If you pass `NULL`, no activation is applied
    (ie. "linear" activation: `a(x) = x`).
@param recurrent_activation Activation function to use
    for the recurrent step.
    Default: sigmoid (`sigmoid`).
    If you pass `NULL`, no activation is applied
    (ie. "linear" activation: `a(x) = x`).
@param use_bias Boolean, (default `TRUE`), whether the layer
    should use a bias vector.
@param kernel_initializer Initializer for the `kernel` weights matrix,
    used for the linear transformation of the inputs. Default:
    `"glorot_uniform"`.
@param recurrent_initializer Initializer for the `recurrent_kernel`
    weights matrix, used for the linear transformation of the recurrent
    state. Default: `"orthogonal"`.
@param bias_initializer Initializer for the bias vector. Default: `"zeros"`.
@param unit_forget_bias Boolean (default `TRUE`). If `TRUE`,
    add 1 to the bias of the forget gate at initialization.
    Setting it to `TRUE` will also force `bias_initializer="zeros"`.
    This is recommended in [Jozefowicz et al.](
    https://github.com/mlresearch/v37/blob/gh-pages/jozefowicz15.pdf)
@param kernel_regularizer Regularizer function applied to the `kernel` weights
    matrix. Default: `NULL`.
@param recurrent_regularizer Regularizer function applied to the
    `recurrent_kernel` weights matrix. Default: `NULL`.
@param bias_regularizer Regularizer function applied to the bias vector.
    Default: `NULL`.
@param activity_regularizer Regularizer function applied to the output of the
    layer (its "activation"). Default: `NULL`.
@param kernel_constraint Constraint function applied to the `kernel` weights
    matrix. Default: `NULL`.
@param recurrent_constraint Constraint function applied to the
    `recurrent_kernel` weights matrix. Default: `NULL`.
@param bias_constraint Constraint function applied to the bias vector.
    Default: `NULL`.
@param dropout Float between 0 and 1. Fraction of the units to drop for the
    linear transformation of the inputs. Default: 0.
@param recurrent_dropout Float between 0 and 1. Fraction of the units to drop
    for the linear transformation of the recurrent state. Default: 0.
@param seed Random seed for dropout.
@param return_sequences Boolean. Whether to return the last output
    in the output sequence, or the full sequence. Default: `FALSE`.
@param return_state Boolean. Whether to return the last state in addition
    to the output. Default: `FALSE`.
@param go_backwards Boolean (default: `FALSE`).
    If `TRUE`, process the input sequence backwards and return the
    reversed sequence.
@param stateful Boolean (default: `FALSE`). If `TRUE`, the last state
    for each sample at index i in a batch will be used as initial
    state for the sample of index i in the following batch.
@param unroll Boolean (default `FALSE`).
    If `TRUE`, the network will be unrolled,
    else a symbolic loop will be used.
    Unrolling can speed-up a RNN,
    although it tends to be more memory-intensive.
    Unrolling is only suitable for short sequences.
@param object Object to compose the layer with. A tensor, array, or sequential model.
@param ... Passed on to the Python callable

@export
@family recurrent layers
@seealso
+ <https:/keras.io/api/layers/recurrent_layers/lstm#lstm-class>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM>
