Cell class for the GRU layer.

@description
This class processes one step within the whole time sequence input, whereas
`keras.layer.GRU` processes the whole sequence.

# Call Arguments
- `inputs`: A 2D tensor, with shape `(batch, features)`.
- `states`: A 2D tensor with shape `(batch, units)`, which is the state
    from the previous time step.
- `training`: Python boolean indicating whether the layer should behave in
    training mode or in inference mode. Only relevant when `dropout` or
    `recurrent_dropout` is used.

# Examples

```r
inputs <- random_uniform(c(32, 10, 8))
outputs <- inputs |> layer_rnn(layer_gru_cell(4))
outputs$shape
```

```
## TensorShape([32, 4])
```

```r
rnn <- layer_rnn(
   cell = layer_gru_cell(4),
   return_sequences=TRUE,
   return_state=TRUE)
c(whole_sequence_output, final_state) %<-% rnn(inputs)
whole_sequence_output$shape
```

```
## TensorShape([32, 10, 4])
```

```r
final_state$shape
```

```
## TensorShape([32, 4])
```

@param units
Positive integer, dimensionality of the output space.

@param activation
Activation function to use. Default: hyperbolic tangent
(`tanh`). If you pass `NULL`, no activation is applied
(ie. "linear" activation: `a(x) = x`).

@param recurrent_activation
Activation function to use for the recurrent step.
Default: sigmoid (`sigmoid`). If you pass `NULL`, no activation is
applied (ie. "linear" activation: `a(x) = x`).

@param use_bias
Boolean, (default `TRUE`), whether the layer
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
matrix. Default: `NULL`.

@param recurrent_regularizer
Regularizer function applied to the
`recurrent_kernel` weights matrix. Default: `NULL`.

@param bias_regularizer
Regularizer function applied to the bias vector.
Default: `NULL`.

@param kernel_constraint
Constraint function applied to the `kernel` weights
matrix. Default: `NULL`.

@param recurrent_constraint
Constraint function applied to the
`recurrent_kernel` weights matrix. Default: `NULL`.

@param bias_constraint
Constraint function applied to the bias vector.
Default: `NULL`.

@param dropout
Float between 0 and 1. Fraction of the units to drop for the
linear transformation of the inputs. Default: 0.

@param recurrent_dropout
Float between 0 and 1. Fraction of the units to drop
for the linear transformation of the recurrent state. Default: 0.

@param reset_after
GRU convention (whether to apply reset gate after or
before matrix multiplication). `FALSE` = `"before"`,
`TRUE` = `"after"` (default and cuDNN compatible).

@param seed
Random seed for dropout.

@param ...
Passed on to the Python callable

@export
@family gru rnn layers
@family rnn layers
@family layers
@seealso
+ <https://www.tensorflow.org/api_docs/python/tf/keras/layers/GRUCell>
