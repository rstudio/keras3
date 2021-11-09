#' Base class for recurrent layers
#'
#' @details
#' See [the Keras RNN API guide](https://www.tensorflow.org/guide/keras/rnn)
#' for details about the usage of RNN API.
#'
#' @inheritParams layer_dense
#'
#' @param cell A RNN cell instance or a list of RNN cell instances.
#' A RNN cell is a class that has:
#' - A `call(input_at_t, states_at_t)` method, returning
#'   `(output_at_t, states_at_t_plus_1)`. The call method of the
#'   cell can also take the optional argument `constants`, see
#'   section "Note on passing external constants" below.
#' - A `state_size` attribute. This can be a single integer
#'   (single state) in which case it is the size of the recurrent
#'   state. This can also be a list of integers (one size per state).
#'   The `state_size` can also be TensorShape or list of
#'   TensorShape, to represent high dimension state.
#' - A `output_size` attribute. This can be a single integer or a
#'   TensorShape, which represent the shape of the output. For backward
#'   compatible reason, if this attribute is not available for the
#'   cell, the value will be inferred by the first element of the
#'   `state_size`.
#' - A `get_initial_state(inputs=NULL, batch_size=NULL, dtype=NULL)`
#'   method that creates a tensor meant to be fed to `call()` as the
#'   initial state, if the user didn't specify any initial state via other
#'   means. The returned initial state should have a shape of
#'   `[batch_size, cell$state_size]`. The cell might choose to create a
#'   tensor full of zeros, or full of other values based on the cell's
#'   implementation.
#'   `inputs` is the input tensor to the RNN layer, which should
#'   contain the batch size as first dimension (`inputs$shape[1]`),
#'   and also dtype (`inputs$dtype`). Note that
#'   the `shape[1]` might be `NULL` during the graph construction. Either
#'   the `inputs` or the pair of `batch_size` and `dtype` are provided.
#'   `batch_size` is a scalar tensor that represents the batch size
#'   of the inputs. `dtype` is `tf.DType` that represents the dtype of
#'   the inputs.
#'   For backward compatibility, if this method is not implemented
#'   by the cell, the RNN layer will create a zero filled tensor with the
#'   size of `[batch_size, cell$state_size]`.
#' In the case that `cell` is a list of RNN cell instances, the cells
#' will be stacked on top of each other in the RNN, resulting in an
#' efficient stacked RNN.
#'
#' @param return_sequences Boolean (default `FALSE`). Whether to return the last
#' output in the output sequence, or the full sequence.
#'
#' @param return_state Boolean (default `FALSE`). Whether to return the last state
#' in addition to the output.
#'
#' @param go_backwards Boolean (default `FALSE`).
#' If `TRUE`, process the input sequence backwards and return the
#' reversed sequence.
#'
#' @param stateful Boolean (default `FALSE`). If `TRUE`, the last state
#' for each sample at index `i` in a batch will be used as initial
#' state for the sample of index `i` in the following batch.
#'
#' @param unroll Boolean (default `FALSE`).
#' If TRUE, the network will be unrolled, else a symbolic loop will be used.
#' Unrolling can speed-up a RNN, although it tends to be more
#' memory-intensive. Unrolling is only suitable for short sequences.
#'
#' @param time_major The shape format of the `inputs` and `outputs` tensors.
#' If `TRUE`, the inputs and outputs will be in shape
#' `(timesteps, batch, ...)`, whereas in the FALSE case, it will be
#' `(batch, timesteps, ...)`. Using `time_major = TRUE` is a bit more
#' efficient because it avoids transposes at the beginning and end of the
#' RNN calculation. However, most TensorFlow data is batch-major, so by
#' default this function accepts input and emits output in batch-major
#' form.
#'
#' @param zero_output_for_mask Boolean (default `FALSE`).
#' Whether the output should use zeros for the masked timesteps. Note that
#' this field is only used when `return_sequences` is TRUE and mask is
#' provided. It can useful if you want to reuse the raw output sequence of
#' the RNN without interference from the masked timesteps, eg, merging
#' bidirectional RNNs.
#'
#' @param ... standard layer arguments.
#'
#' @section Call arguments:
#'   - `inputs`: Input tensor.
#'   - `mask`: Binary tensor of shape `[batch_size, timesteps]` indicating whether
#'     a given timestep should be masked. An individual `TRUE` entry indicates
#'     that the corresponding timestep should be utilized, while a `FALSE`
#'     entry indicates that the corresponding timestep should be ignored.
#'   - `training`: R or Python Boolean indicating whether the layer should behave in
#'     training mode or in inference mode. This argument is passed to the cell
#'     when calling it. This is for use with cells that use dropout.
#'   - `initial_state`: List of initial state tensors to be passed to the first
#'     call of the cell.
#'   - `constants`: List of constant tensors to be passed to the cell at each
#'     timestep.
#'
#' @template roxlate-recurrent-layer
#'
#' @seealso
#'   +  <https://www.tensorflow.org/api_docs/python/tf/keras/layers/RNN>
#'   +  <https://keras.io/api/layers/recurrent_layers/rnn>
#'   +  `reticulate::py_help(keras$layers$RNN)`
#'
#' @export
layer_rnn <-
function(object, cell,
         return_sequences = FALSE,
         return_state = FALSE,
         go_backwards = FALSE,
         stateful = FALSE,
         unroll = FALSE,
         time_major = FALSE,
         ...,
         zero_output_for_mask = FALSE)
{
    args <- capture_args(match.call(), ignore = "object")
    create_layer(keras$layers$RNN, object, args)
}


#' Cell class for SimpleRNN
#'
#' @details
#' See [the Keras RNN API guide](https://www.tensorflow.org/guide/keras/rnn)
#' for details about the usage of RNN API.
#'
#' This class processes one step within the whole time sequence input, whereas
#' `tf.keras.layer.SimpleRNN` processes the whole sequence.
#'
#' @param units Positive integer, dimensionality of the output space.
#'
#' @param activation Activation function to use.
#' Default: hyperbolic tangent (`tanh`).
#' If you pass `NULL`, no activation is applied
#' (ie. "linear" activation: `a(x) = x`).
#'
#' @param use_bias Boolean, (default `TRUE`), whether the layer uses a bias vector.
#'
#' @param kernel_initializer Initializer for the `kernel` weights matrix,
#' used for the linear transformation of the inputs. Default:
#' `glorot_uniform`.
#'
#' @param recurrent_initializer Initializer for the `recurrent_kernel`
#' weights matrix, used for the linear transformation of the recurrent state.
#' Default: `orthogonal`.
#'
#' @param bias_initializer Initializer for the bias vector. Default: `zeros`.
#'
#' @param kernel_regularizer Regularizer function applied to the `kernel` weights
#' matrix. Default: `NULL`.
#'
#' @param recurrent_regularizer Regularizer function applied to the
#' `recurrent_kernel` weights matrix. Default: `NULL`.
#'
#' @param bias_regularizer Regularizer function applied to the bias vector. Default:
#' `NULL`.
#'
#' @param kernel_constraint Constraint function applied to the `kernel` weights
#' matrix. Default: `NULL`.
#'
#' @param recurrent_constraint Constraint function applied to the `recurrent_kernel`
#' weights matrix. Default: `NULL`.
#'
#' @param bias_constraint Constraint function applied to the bias vector. Default:
#' `NULL`.
#'
#' @param dropout Float between 0 and 1. Fraction of the units to drop for the linear
#' transformation of the inputs. Default: 0.
#'
#' @param recurrent_dropout Float between 0 and 1. Fraction of the units to drop for
#' the linear transformation of the recurrent state. Default: 0.
#'
#' @param ... standard layer arguments.
#'
#' @family RNN cell layers
#'
#' @seealso
#'   +  <https://www.tensorflow.org/api_docs/python/tf/keras/layers/SimpleRNNCell>
#'   +  <https://keras.io/api/layers>
#' @export
layer_simple_rnn_cell <-
function(units,
         activation = "tanh",
         use_bias = TRUE,
         kernel_initializer = "glorot_uniform",
         recurrent_initializer = "orthogonal",
         bias_initializer = "zeros",
         kernel_regularizer = NULL,
         recurrent_regularizer = NULL,
         bias_regularizer = NULL,
         kernel_constraint = NULL,
         recurrent_constraint = NULL,
         bias_constraint = NULL,
         dropout = 0,
         recurrent_dropout = 0,
         ...)
{
    args <- capture_args(match.call(), list(units = as.integer))
    do.call(keras$layers$SimpleRNNCell, args)
}




#' Cell class for the GRU layer
#'
#' @details
#' See [the Keras RNN API guide](https://www.tensorflow.org/guide/keras/rnn)
#' for details about the usage of RNN API.
#'
#' This class processes one step within the whole time sequence input, whereas
#' `tf.keras.layer.GRU` processes the whole sequence.
#'
#' For example:
#'  ````r
#'  inputs <- k_random_uniform(c(32, 10, 8))
#'  output <- inputs %>% layer_rnn(layer_gru_cell(4))
#'  output$shape  # TensorShape([32, 4])
#'
#'  rnn <- layer_rnn(cell = layer_gru_cell(4),
#'                   return_sequence = TRUE,
#'                   return_state = TRUE)
#'  c(whole_sequence_output, final_state) %<-% rnn(inputs)
#'  whole_sequence_output$shape # TensorShape([32, 10, 4])
#'  final_state$shape           # TensorShape([32, 4])
#'  ````
#'
#' @param units Positive integer, dimensionality of the output space.
#'
#' @param activation Activation function to use. Default: hyperbolic tangent
#' (`tanh`). If you pass `NULL`, no activation is applied
#' (ie. "linear" activation: `a(x) = x`).
#'
#' @param recurrent_activation Activation function to use for the recurrent step.
#' Default: sigmoid (`sigmoid`). If you pass `NULL`, no activation is
#' applied (ie. "linear" activation: `a(x) = x`).
#'
#' @param use_bias Boolean, (default `TRUE`), whether the layer uses a bias vector.
#'
#' @param kernel_initializer Initializer for the `kernel` weights matrix,
#' used for the linear transformation of the inputs. Default:
#' `glorot_uniform`.
#'
#' @param recurrent_initializer Initializer for the `recurrent_kernel`
#' weights matrix, used for the linear transformation of the recurrent state.
#' Default: `orthogonal`.
#'
#' @param bias_initializer Initializer for the bias vector. Default: `zeros`.
#'
#' @param kernel_regularizer Regularizer function applied to the `kernel` weights
#' matrix. Default: `NULL`.
#'
#' @param recurrent_regularizer Regularizer function applied to the
#' `recurrent_kernel` weights matrix. Default: `NULL`.
#'
#' @param bias_regularizer Regularizer function applied to the bias vector. Default:
#' `NULL`.
#'
#' @param kernel_constraint Constraint function applied to the `kernel` weights
#' matrix. Default: `NULL`.
#'
#' @param recurrent_constraint Constraint function applied to the `recurrent_kernel`
#' weights matrix. Default: `NULL`.
#'
#' @param bias_constraint Constraint function applied to the bias vector. Default:
#' `NULL`.
#'
#' @param dropout Float between 0 and 1. Fraction of the units to drop for the
#' linear transformation of the inputs. Default: 0.
#'
#' @param recurrent_dropout Float between 0 and 1. Fraction of the units to drop for
#' the linear transformation of the recurrent state. Default: 0.
#'
#' @param reset_after GRU convention (whether to apply reset gate after or
#' before matrix multiplication). FALSE = "before",
#' TRUE = "after" (default and CuDNN compatible).
#' @param ... standard layer arguments.
#'
#' @family RNN cell layers
#'
#' @seealso
#'   +  <https://www.tensorflow.org/api_docs/python/tf/keras/layers/GRUCell>
#'
#' @export
layer_gru_cell <-
function(units,
         activation = "tanh",
         recurrent_activation = "sigmoid",
         use_bias = TRUE,
         kernel_initializer = "glorot_uniform",
         recurrent_initializer = "orthogonal",
         bias_initializer = "zeros",
         kernel_regularizer = NULL,
         recurrent_regularizer = NULL,
         bias_regularizer = NULL,
         kernel_constraint = NULL,
         recurrent_constraint = NULL,
         bias_constraint = NULL,
         dropout = 0,
         recurrent_dropout = 0,
         reset_after = TRUE,
         ...)
{
    args <- capture_args(match.call(), list(units = as.integer))
    do.call(keras$layers$GRUCell, args)
}


#' Wrapper allowing a stack of RNN cells to behave as a single cell
#'
#' Used to implement efficient stacked RNNs.
#'
#' @param cells List of RNN cell instances.
#' @param ... standard layer arguments.
#'
#' @family RNN cell layers
#'
#' @seealso
#'   +  <https://www.tensorflow.org/api_docs/python/tf/keras/layers/StackedRNNCells>
#'
#' @export
layer_stacked_rnn_cells <-
function(cells, ...)
{
    args <- capture_args(match.call())
    do.call(keras$layers$StackedRNNCells, args)
}


#' Cell class for the LSTM layer
#'
#' @details
#' See [the Keras RNN API guide](https://www.tensorflow.org/guide/keras/rnn)
#' for details about the usage of RNN API.
#'
#' This class processes one step within the whole time sequence input, whereas
#' `tf$keras$layer$LSTM` processes the whole sequence.
#'
#' For example:
#' ````r
#' inputs <- k_random_normal(c(32, 10, 8))
#' rnn <- layer_rnn(cell = layer_lstm_cell(units = 4))
#' output <- rnn(inputs)
#' dim(output) # (32, 4)
#'
#' rnn <- layer_rnn(cell = layer_lstm_cell(units = 4),
#'                  return_sequences = TRUE,
#'                  return_state = TRUE)
#' c(whole_seq_output, final_memory_state, final_carry_state) %<-% rnn(inputs)
#'
#' dim(whole_seq_output)   # (32, 10, 4)
#' dim(final_memory_state) # (32, 4)
#' dim(final_carry_state)  # (32, 4)
#' ````
#'
#' @param units Positive integer, dimensionality of the output space.
#'
#' @param activation Activation function to use. Default: hyperbolic tangent
#' (`tanh`). If you pass `NULL`, no activation is applied (ie. "linear"
#' activation: `a(x) = x`).
#'
#' @param recurrent_activation Activation function to use for the recurrent step.
#' Default: sigmoid (`sigmoid`). If you pass `NULL`, no activation is applied
#' (ie. "linear" activation: `a(x) = x`).
#'
#' @param use_bias Boolean, (default `TRUE`), whether the layer uses a bias vector.
#'
#' @param kernel_initializer Initializer for the `kernel` weights matrix, used for
#' the linear transformation of the inputs. Default: `glorot_uniform`.
#'
#' @param recurrent_initializer Initializer for the `recurrent_kernel` weights
#' matrix, used for the linear transformation of the recurrent state.
#' Default: `orthogonal`.
#'
#' @param bias_initializer Initializer for the bias vector. Default: `zeros`.
#'
#' @param unit_forget_bias Boolean (default `TRUE`). If TRUE, add 1 to the bias of
#' the forget gate at initialization. Setting it to true will also force
#' `bias_initializer="zeros"`. This is recommended in [Jozefowicz et
#'   al.](https://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
#'
#' @param kernel_regularizer Regularizer function applied to the `kernel` weights
#' matrix. Default: `NULL`.
#'
#' @param recurrent_regularizer Regularizer function applied to
#' the `recurrent_kernel` weights matrix. Default: `NULL`.
#'
#' @param bias_regularizer Regularizer function applied to the bias vector. Default:
#' `NULL`.
#'
#' @param kernel_constraint Constraint function applied to the `kernel` weights
#' matrix. Default: `NULL`.
#'
#' @param recurrent_constraint Constraint function applied to the `recurrent_kernel`
#' weights matrix. Default: `NULL`.
#'
#' @param bias_constraint Constraint function applied to the bias vector. Default:
#' `NULL`.
#'
#' @param dropout Float between 0 and 1. Fraction of the units to drop for the linear
#' transformation of the inputs. Default: 0.
#'
#' @param recurrent_dropout Float between 0 and 1. Fraction of the units to drop for
#' the linear transformation of the recurrent state. Default: 0.
#'
#' @param ... standard layer arguments.
#'
#' @family RNN cell layers
#'
#' @seealso
#'   +  <https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTMCell>
#'
#' @export
layer_lstm_cell <-
function(units,
         activation = "tanh",
         recurrent_activation = "sigmoid",
         use_bias = TRUE,
         kernel_initializer = "glorot_uniform",
         recurrent_initializer = "orthogonal",
         bias_initializer = "zeros",
         unit_forget_bias = TRUE,
         kernel_regularizer = NULL,
         recurrent_regularizer = NULL,
         bias_regularizer = NULL,
         kernel_constraint = NULL,
         recurrent_constraint = NULL,
         bias_constraint = NULL,
         dropout = 0,
         recurrent_dropout = 0,
         ...)
{
    args <- capture_args(match.call(), list(units = as.integer))
    do.call(keras$layers$LSTMCell, args)
}
