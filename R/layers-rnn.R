

#' Bidirectional wrapper for RNNs.
#'
#' @description
#'
#' # Call Arguments
#' The call arguments for this layer are the same as those of the
#' wrapped RNN layer. Beware that when passing the `initial_state`
#' argument during the call of this layer, the first half in the
#' list of elements in the `initial_state` list will be passed to
#' the forward RNN call and the last half in the list of elements
#' will be passed to the backward RNN call.
#'
#' # Note
#' instantiating a `Bidirectional` layer from an existing RNN layer
#' instance will not reuse the weights state of the RNN layer instance -- the
#' `Bidirectional` layer will have freshly initialized weights.
#'
#' # Examples
#' ```{r}
#' model <- keras_model_sequential(input_shape = c(5, 10)) %>%
#'   bidirectional(layer_lstm(units = 10, return_sequences = TRUE)) %>%
#'   bidirectional(layer_lstm(units = 10)) %>%
#'   layer_dense(5, activation = "softmax")
#'
#' model %>% compile(loss = "categorical_crossentropy",
#'                   optimizer = "rmsprop")
#'
#' # With custom backward layer
#' forward_layer <- layer_lstm(units = 10, return_sequences = TRUE)
#' backward_layer <- layer_lstm(units = 10, activation = "relu",
#'                              return_sequences = TRUE, go_backwards = TRUE)
#'
#' model <- keras_model_sequential(input_shape = c(5, 10)) %>%
#'   bidirectional(forward_layer, backward_layer = backward_layer) %>%
#'   layer_dense(5, activation = "softmax")
#'
#' model %>% compile(loss = "categorical_crossentropy",
#'                   optimizer = "rmsprop")
#' ```
#'
#' @param layer
#' `keras.layers.RNN` instance, such as
#' `keras.layers.LSTM` or `keras.layers.GRU`.
#' It could also be a `keras.layers.Layer` instance
#' that meets the following criteria:
#' 1. Be a sequence-processing layer (accepts 3D+ inputs).
#' 2. Have a `go_backwards`, `return_sequences` and `return_state`
#' attribute (with the same semantics as for the `RNN` class).
#' 3. Have an `input_spec` attribute.
#' 4. Implement serialization via `get_config()` and `from_config()`.
#' Note that the recommended way to create new RNN layers is to write a
#' custom RNN cell and use it with `keras.layers.RNN`, instead of
#' subclassing `keras.layers.Layer` directly.
#' When `return_sequences` is `TRUE`, the output of the masked
#' timestep will be zero regardless of the layer's original
#' `zero_output_for_mask` value.
#'
#' @param merge_mode
#' Mode by which outputs of the forward and backward RNNs
#' will be combined. One of `{"sum", "mul", "concat", "ave", NULL}`.
#' If `NULL`, the outputs will not be combined,
#' they will be returned as a list. Defaults to `"concat"`.
#'
#' @param backward_layer
#' Optional `keras.layers.RNN`,
#' or `keras.layers.Layer` instance to be used to handle
#' backwards input processing.
#' If `backward_layer` is not provided, the layer instance passed
#' as the `layer` argument will be used to generate the backward layer
#' automatically.
#' Note that the provided `backward_layer` layer should have properties
#' matching those of the `layer` argument, in particular
#' it should have the same values for `stateful`, `return_states`,
#' `return_sequences`, etc. In addition, `backward_layer`
#' and `layer` should have different `go_backwards` argument values.
#' A `ValueError` will be raised if these requirements are not met.
#'
#' @param object
#' Object to compose the layer with. A tensor, array, or sequential model.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @param weights
#' see description
#'
#' @export
#' @family rnn layers
#' @family layers
#' @seealso
#' + <https:/keras.io/api/layers/recurrent_layers/bidirectional#bidirectional-class>
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Bidirectional>
#' @tether keras.layers.Bidirectional
bidirectional <-
function (object, layer, merge_mode = "concat", weights = NULL,
    backward_layer = NULL, ...)
{
    args <- capture_args2(list(input_shape = normalize_shape,
        batch_size = as_integer, batch_input_shape = normalize_shape),
        ignore = "object")
    create_layer(keras$layers$Bidirectional, object, args)
}


#' 1D Convolutional LSTM.
#'
#' @description
#' Similar to an LSTM layer, but the input transformations
#' and recurrent transformations are both convolutional.
#'
#' # Call Arguments
#' - `inputs`: A 4D tensor.
#' - `initial_state`: List of initial state tensors to be passed to the first
#'     call of the cell.
#' - `mask`: Binary tensor of shape `(samples, timesteps)` indicating whether a
#'     given timestep should be masked.
#' - `training`: Python boolean indicating whether the layer should behave in
#'     training mode or in inference mode.
#'     This is only relevant if `dropout` or `recurrent_dropout` are set.
#'
#' # Input Shape
#' - If `data_format="channels_first"`:
#'     4D tensor with shape: `(samples, time, channels, rows)`
#' - If `data_format="channels_last"`:
#'     4D tensor with shape: `(samples, time, rows, channels)`
#'
#' # Output Shape
#' - If `return_state`: a list of tensors. The first tensor is the output.
#'     The remaining tensors are the last states,
#'     each 3D tensor with shape: `(samples, filters, new_rows)` if
#'     `data_format='channels_first'`
#'     or shape: `(samples, new_rows, filters)` if
#'     `data_format='channels_last'`.
#'     `rows` values might have changed due to padding.
#' - If `return_sequences`: 4D tensor with shape: `(samples, timesteps,
#'     filters, new_rows)` if data_format='channels_first'
#'     or shape: `(samples, timesteps, new_rows, filters)` if
#'     `data_format='channels_last'`.
#' - Else, 3D tensor with shape: `(samples, filters, new_rows)` if
#'     `data_format='channels_first'`
#'     or shape: `(samples, new_rows, filters)` if
#'     `data_format='channels_last'`.
#'
#' # References
#' - [Shi et al., 2015](http://arxiv.org/abs/1506.04214v1)
#'     (the current implementation does not include the feedback loop on the
#'     cells output).
#'
#' @param filters
#' int, the dimension of the output space (the number of filters
#' in the convolution).
#'
#' @param kernel_size
#' int or tuple/list of 1 integer, specifying the size of
#' the convolution window.
#'
#' @param strides
#' int or tuple/list of 1 integer, specifying the stride length
#' of the convolution. `strides > 1` is incompatible with
#' `dilation_rate > 1`.
#'
#' @param padding
#' string, `"valid"` or `"same"` (case-insensitive).
#' `"valid"` means no padding. `"same"` results in padding evenly to
#' the left/right or up/down of the input such that output has the
#' same height/width dimension as the input.
#'
#' @param data_format
#' string, either `"channels_last"` or `"channels_first"`.
#' The ordering of the dimensions in the inputs. `"channels_last"`
#' corresponds to inputs with shape `(batch, steps, features)`
#' while `"channels_first"` corresponds to inputs with shape
#' `(batch, features, steps)`. It defaults to the `image_data_format`
#' value found in your Keras config file at `~/.keras/keras.json`.
#' If you never set it, then it will be `"channels_last"`.
#'
#' @param dilation_rate
#' int or tuple/list of 1 integers, specifying the dilation
#' rate to use for dilated convolution.
#'
#' @param activation
#' Activation function to use. By default hyperbolic tangent
#' activation function is applied (`tanh(x)`).
#'
#' @param recurrent_activation
#' Activation function to use for the recurrent step.
#'
#' @param use_bias
#' Boolean, whether the layer uses a bias vector.
#'
#' @param kernel_initializer
#' Initializer for the `kernel` weights matrix,
#' used for the linear transformation of the inputs.
#'
#' @param recurrent_initializer
#' Initializer for the `recurrent_kernel` weights
#' matrix, used for the linear transformation of the recurrent state.
#'
#' @param bias_initializer
#' Initializer for the bias vector.
#'
#' @param unit_forget_bias
#' Boolean. If `TRUE`, add 1 to the bias of
#' the forget gate at initialization.
#' Use in combination with `bias_initializer="zeros"`.
#' This is recommended in [Jozefowicz et al., 2015](
#' http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
#'
#' @param kernel_regularizer
#' Regularizer function applied to the `kernel` weights
#' matrix.
#'
#' @param recurrent_regularizer
#' Regularizer function applied to the
#' `recurrent_kernel` weights matrix.
#'
#' @param bias_regularizer
#' Regularizer function applied to the bias vector.
#'
#' @param activity_regularizer
#' Regularizer function applied to.
#'
#' @param kernel_constraint
#' Constraint function applied to the `kernel` weights
#' matrix.
#'
#' @param recurrent_constraint
#' Constraint function applied to the
#' `recurrent_kernel` weights matrix.
#'
#' @param bias_constraint
#' Constraint function applied to the bias vector.
#'
#' @param dropout
#' Float between 0 and 1. Fraction of the units to drop for the
#' linear transformation of the inputs.
#'
#' @param recurrent_dropout
#' Float between 0 and 1. Fraction of the units to drop
#' for the linear transformation of the recurrent state.
#'
#' @param seed
#' Random seed for dropout.
#'
#' @param return_sequences
#' Boolean. Whether to return the last output
#' in the output sequence, or the full sequence. Default: `FALSE`.
#'
#' @param return_state
#' Boolean. Whether to return the last state in addition
#' to the output. Default: `FALSE`.
#'
#' @param go_backwards
#' Boolean (default: `FALSE`).
#' If `TRUE`, process the input sequence backwards and return the
#' reversed sequence.
#'
#' @param stateful
#' Boolean (default `FALSE`). If `TRUE`, the last state
#' for each sample at index i in a batch will be used as initial
#' state for the sample of index i in the following batch.
#'
#' @param unroll
#' Boolean (default: `FALSE`).
#' If `TRUE`, the network will be unrolled,
#' else a symbolic loop will be used.
#' Unrolling can speed-up a RNN,
#' although it tends to be more memory-intensive.
#' Unrolling is only suitable for short sequences.
#'
#' @param object
#' Object to compose the layer with. A tensor, array, or sequential model.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @export
#' @family rnn layers
#' @family layers
#' @seealso
#' + <https:/keras.io/api/layers/recurrent_layers/conv_lstm1d#convlstm1d-class>
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/layers/ConvLSTM1D>
#' @tether keras.layers.ConvLSTM1D
layer_conv_lstm_1d <-
function (object, filters, kernel_size, strides = 1L, padding = "valid",
    data_format = NULL, dilation_rate = 1L, activation = "tanh",
    recurrent_activation = "sigmoid", use_bias = TRUE, kernel_initializer = "glorot_uniform",
    recurrent_initializer = "orthogonal", bias_initializer = "zeros",
    unit_forget_bias = TRUE, kernel_regularizer = NULL, recurrent_regularizer = NULL,
    bias_regularizer = NULL, activity_regularizer = NULL, kernel_constraint = NULL,
    recurrent_constraint = NULL, bias_constraint = NULL, dropout = 0,
    recurrent_dropout = 0, seed = NULL, return_sequences = FALSE,
    return_state = FALSE, go_backwards = FALSE, stateful = FALSE,
    ..., unroll = NULL)
{
    args <- capture_args2(list(filters = as_integer, kernel_size = as_integer_tuple,
        strides = as_integer_tuple, dilation_rate = as_integer_tuple,
        seed = as_integer, input_shape = normalize_shape, batch_size = as_integer,
        batch_input_shape = normalize_shape), ignore = "object")
    create_layer(keras$layers$ConvLSTM1D, object, args)
}


#' 2D Convolutional LSTM.
#'
#' @description
#' Similar to an LSTM layer, but the input transformations
#' and recurrent transformations are both convolutional.
#'
#' # Call Arguments
#' - `inputs`: A 5D tensor.
#' - `mask`: Binary tensor of shape `(samples, timesteps)` indicating whether a
#'     given timestep should be masked.
#' - `training`: Python boolean indicating whether the layer should behave in
#'     training mode or in inference mode.
#'     This is only relevant if `dropout` or `recurrent_dropout` are set.
#' - `initial_state`: List of initial state tensors to be passed to the first
#'     call of the cell.
#'
#' # Input Shape
#' - If `data_format='channels_first'`:
#'     5D tensor with shape: `(samples, time, channels, rows, cols)`
#' - If `data_format='channels_last'`:
#'     5D tensor with shape: `(samples, time, rows, cols, channels)`
#'
#' # Output Shape
#' - If `return_state`: a list of tensors. The first tensor is the output.
#'     The remaining tensors are the last states,
#'     each 4D tensor with shape: `(samples, filters, new_rows, new_cols)` if
#'     `data_format='channels_first'`
#'     or shape: `(samples, new_rows, new_cols, filters)` if
#'     `data_format='channels_last'`. `rows` and `cols` values might have
#'     changed due to padding.
#' - If `return_sequences`: 5D tensor with shape: `(samples, timesteps,
#'     filters, new_rows, new_cols)` if data_format='channels_first'
#'     or shape: `(samples, timesteps, new_rows, new_cols, filters)` if
#'     `data_format='channels_last'`.
#' - Else, 4D tensor with shape: `(samples, filters, new_rows, new_cols)` if
#'     `data_format='channels_first'`
#'     or shape: `(samples, new_rows, new_cols, filters)` if
#'     `data_format='channels_last'`.
#'
#' # References
#' - [Shi et al., 2015](http://arxiv.org/abs/1506.04214v1)
#'     (the current implementation does not include the feedback loop on the
#'     cells output).
#'
#' @param filters
#' int, the dimension of the output space (the number of filters
#' in the convolution).
#'
#' @param kernel_size
#' int or tuple/list of 2 integers, specifying the size of the
#' convolution window.
#'
#' @param strides
#' int or tuple/list of 2 integers, specifying the stride length
#' of the convolution. `strides > 1` is incompatible with
#' `dilation_rate > 1`.
#'
#' @param padding
#' string, `"valid"` or `"same"` (case-insensitive).
#' `"valid"` means no padding. `"same"` results in padding evenly to
#' the left/right or up/down of the input such that output has the same
#' height/width dimension as the input.
#'
#' @param data_format
#' string, either `"channels_last"` or `"channels_first"`.
#' The ordering of the dimensions in the inputs. `"channels_last"`
#' corresponds to inputs with shape `(batch, steps, features)`
#' while `"channels_first"` corresponds to inputs with shape
#' `(batch, features, steps)`. It defaults to the `image_data_format`
#' value found in your Keras config file at `~/.keras/keras.json`.
#' If you never set it, then it will be `"channels_last"`.
#'
#' @param dilation_rate
#' int or tuple/list of 2 integers, specifying the dilation
#' rate to use for dilated convolution.
#'
#' @param activation
#' Activation function to use. By default hyperbolic tangent
#' activation function is applied (`tanh(x)`).
#'
#' @param recurrent_activation
#' Activation function to use for the recurrent step.
#'
#' @param use_bias
#' Boolean, whether the layer uses a bias vector.
#'
#' @param kernel_initializer
#' Initializer for the `kernel` weights matrix,
#' used for the linear transformation of the inputs.
#'
#' @param recurrent_initializer
#' Initializer for the `recurrent_kernel` weights
#' matrix, used for the linear transformation of the recurrent state.
#'
#' @param bias_initializer
#' Initializer for the bias vector.
#'
#' @param unit_forget_bias
#' Boolean. If `TRUE`, add 1 to the bias of the forget
#' gate at initialization.
#' Use in combination with `bias_initializer="zeros"`.
#' This is recommended in [Jozefowicz et al., 2015](
#' http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
#'
#' @param kernel_regularizer
#' Regularizer function applied to the `kernel` weights
#' matrix.
#'
#' @param recurrent_regularizer
#' Regularizer function applied to the
#' `recurrent_kernel` weights matrix.
#'
#' @param bias_regularizer
#' Regularizer function applied to the bias vector.
#'
#' @param activity_regularizer
#' Regularizer function applied to.
#'
#' @param kernel_constraint
#' Constraint function applied to the `kernel` weights
#' matrix.
#'
#' @param recurrent_constraint
#' Constraint function applied to the
#' `recurrent_kernel` weights matrix.
#'
#' @param bias_constraint
#' Constraint function applied to the bias vector.
#'
#' @param dropout
#' Float between 0 and 1. Fraction of the units to drop for the
#' linear transformation of the inputs.
#'
#' @param recurrent_dropout
#' Float between 0 and 1. Fraction of the units to drop
#' for the linear transformation of the recurrent state.
#'
#' @param seed
#' Random seed for dropout.
#'
#' @param return_sequences
#' Boolean. Whether to return the last output
#' in the output sequence, or the full sequence. Default: `FALSE`.
#'
#' @param return_state
#' Boolean. Whether to return the last state in addition
#' to the output. Default: `FALSE`.
#'
#' @param go_backwards
#' Boolean (default: `FALSE`).
#' If `TRUE`, process the input sequence backwards and return the
#' reversed sequence.
#'
#' @param stateful
#' Boolean (default FALSE). If `TRUE`, the last state
#' for each sample at index i in a batch will be used as initial
#' state for the sample of index i in the following batch.
#'
#' @param unroll
#' Boolean (default: `FALSE`).
#' If `TRUE`, the network will be unrolled,
#' else a symbolic loop will be used.
#' Unrolling can speed-up a RNN,
#' although it tends to be more memory-intensive.
#' Unrolling is only suitable for short sequences.
#'
#' @param object
#' Object to compose the layer with. A tensor, array, or sequential model.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @export
#' @family rnn layers
#' @family layers
#' @seealso
#' + <https:/keras.io/api/layers/recurrent_layers/conv_lstm2d#convlstm2d-class>
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/layers/ConvLSTM2D>
#' @tether keras.layers.ConvLSTM2D
layer_conv_lstm_2d <-
function (object, filters, kernel_size, strides = 1L, padding = "valid",
    data_format = NULL, dilation_rate = 1L, activation = "tanh",
    recurrent_activation = "sigmoid", use_bias = TRUE, kernel_initializer = "glorot_uniform",
    recurrent_initializer = "orthogonal", bias_initializer = "zeros",
    unit_forget_bias = TRUE, kernel_regularizer = NULL, recurrent_regularizer = NULL,
    bias_regularizer = NULL, activity_regularizer = NULL, kernel_constraint = NULL,
    recurrent_constraint = NULL, bias_constraint = NULL, dropout = 0,
    recurrent_dropout = 0, seed = NULL, return_sequences = FALSE,
    return_state = FALSE, go_backwards = FALSE, stateful = FALSE,
    ..., unroll = NULL)
{
    args <- capture_args2(list(filters = as_integer, kernel_size = as_integer_tuple,
        strides = as_integer_tuple, dilation_rate = as_integer_tuple,
        seed = as_integer, input_shape = normalize_shape, batch_size = as_integer,
        batch_input_shape = normalize_shape), ignore = "object")
    create_layer(keras$layers$ConvLSTM2D, object, args)
}


#' 3D Convolutional LSTM.
#'
#' @description
#' Similar to an LSTM layer, but the input transformations
#' and recurrent transformations are both convolutional.
#'
#' # Call Arguments
#' - `inputs`: A 6D tensor.
#' - `mask`: Binary tensor of shape `(samples, timesteps)` indicating whether a
#'     given timestep should be masked.
#' - `training`: Python boolean indicating whether the layer should behave in
#'     training mode or in inference mode.
#'     This is only relevant if `dropout` or `recurrent_dropout` are set.
#' - `initial_state`: List of initial state tensors to be passed to the first
#'     call of the cell.
#'
#' # Input Shape
#' - If `data_format='channels_first'`:
#'     5D tensor with shape: `(samples, time, channels, *spatial_dims)`
#' - If `data_format='channels_last'`:
#'     5D tensor with shape: `(samples, time, *spatial_dims, channels)`
#'
#' # Output Shape
#' - If `return_state`: a list of tensors. The first tensor is the output.
#'     The remaining tensors are the last states,
#'     each 4D tensor with shape: `(samples, filters, *spatial_dims)` if
#'     `data_format='channels_first'`
#'     or shape: `(samples, *spatial_dims, filters)` if
#'     `data_format='channels_last'`.
#' - If `return_sequences`: 5D tensor with shape: `(samples, timesteps,
#'     filters, *spatial_dims)` if data_format='channels_first'
#'     or shape: `(samples, timesteps, *spatial_dims, filters)` if
#'     `data_format='channels_last'`.
#' - Else, 4D tensor with shape: `(samples, filters, *spatial_dims)` if
#'     `data_format='channels_first'`
#'     or shape: `(samples, *spatial_dims, filters)` if
#'     `data_format='channels_last'`.
#'
#' # References
#' - [Shi et al., 2015](http://arxiv.org/abs/1506.04214v1)
#'     (the current implementation does not include the feedback loop on the
#'     cells output).
#'
#' @param filters
#' int, the dimension of the output space (the number of filters
#' in the convolution).
#'
#' @param kernel_size
#' int or tuple/list of 3 integers, specifying the size of the
#' convolution window.
#'
#' @param strides
#' int or tuple/list of 3 integers, specifying the stride length
#' of the convolution. `strides > 1` is incompatible with
#' `dilation_rate > 1`.
#'
#' @param padding
#' string, `"valid"` or `"same"` (case-insensitive).
#' `"valid"` means no padding. `"same"` results in padding evenly to
#' the left/right or up/down of the input such that output has the same
#' height/width dimension as the input.
#'
#' @param data_format
#' string, either `"channels_last"` or `"channels_first"`.
#' The ordering of the dimensions in the inputs. `"channels_last"`
#' corresponds to inputs with shape `(batch, steps, features)`
#' while `"channels_first"` corresponds to inputs with shape
#' `(batch, features, steps)`. It defaults to the `image_data_format`
#' value found in your Keras config file at `~/.keras/keras.json`.
#' If you never set it, then it will be `"channels_last"`.
#'
#' @param dilation_rate
#' int or tuple/list of 3 integers, specifying the dilation
#' rate to use for dilated convolution.
#'
#' @param activation
#' Activation function to use. By default hyperbolic tangent
#' activation function is applied (`tanh(x)`).
#'
#' @param recurrent_activation
#' Activation function to use for the recurrent step.
#'
#' @param use_bias
#' Boolean, whether the layer uses a bias vector.
#'
#' @param kernel_initializer
#' Initializer for the `kernel` weights matrix,
#' used for the linear transformation of the inputs.
#'
#' @param recurrent_initializer
#' Initializer for the `recurrent_kernel` weights
#' matrix, used for the linear transformation of the recurrent state.
#'
#' @param bias_initializer
#' Initializer for the bias vector.
#'
#' @param unit_forget_bias
#' Boolean. If `TRUE`, add 1 to the bias of the forget
#' gate at initialization.
#' Use in combination with `bias_initializer="zeros"`.
#' This is recommended in [Jozefowicz et al., 2015](
#' http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
#'
#' @param kernel_regularizer
#' Regularizer function applied to the `kernel` weights
#' matrix.
#'
#' @param recurrent_regularizer
#' Regularizer function applied to the
#' `recurrent_kernel` weights matrix.
#'
#' @param bias_regularizer
#' Regularizer function applied to the bias vector.
#'
#' @param activity_regularizer
#' Regularizer function applied to.
#'
#' @param kernel_constraint
#' Constraint function applied to the `kernel` weights
#' matrix.
#'
#' @param recurrent_constraint
#' Constraint function applied to the
#' `recurrent_kernel` weights matrix.
#'
#' @param bias_constraint
#' Constraint function applied to the bias vector.
#'
#' @param dropout
#' Float between 0 and 1. Fraction of the units to drop for the
#' linear transformation of the inputs.
#'
#' @param recurrent_dropout
#' Float between 0 and 1. Fraction of the units to drop
#' for the linear transformation of the recurrent state.
#'
#' @param seed
#' Random seed for dropout.
#'
#' @param return_sequences
#' Boolean. Whether to return the last output
#' in the output sequence, or the full sequence. Default: `FALSE`.
#'
#' @param return_state
#' Boolean. Whether to return the last state in addition
#' to the output. Default: `FALSE`.
#'
#' @param go_backwards
#' Boolean (default: `FALSE`).
#' If `TRUE`, process the input sequence backwards and return the
#' reversed sequence.
#'
#' @param stateful
#' Boolean (default `FALSE`). If `TRUE`, the last state
#' for each sample at index i in a batch will be used as initial
#' state for the sample of index i in the following batch.
#'
#' @param unroll
#' Boolean (default: `FALSE`).
#' If `TRUE`, the network will be unrolled,
#' else a symbolic loop will be used.
#' Unrolling can speed-up a RNN,
#' although it tends to be more memory-intensive.
#' Unrolling is only suitable for short sequences.
#'
#' @param object
#' Object to compose the layer with. A tensor, array, or sequential model.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @export
#' @family rnn layers
#' @family layers
#' @seealso
#' + <https:/keras.io/api/layers/recurrent_layers/conv_lstm3d#convlstm3d-class>
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/layers/ConvLSTM3D>
#' @tether keras.layers.ConvLSTM3D
layer_conv_lstm_3d <-
function (object, filters, kernel_size, strides = 1L, padding = "valid",
    data_format = NULL, dilation_rate = 1L, activation = "tanh",
    recurrent_activation = "sigmoid", use_bias = TRUE, kernel_initializer = "glorot_uniform",
    recurrent_initializer = "orthogonal", bias_initializer = "zeros",
    unit_forget_bias = TRUE, kernel_regularizer = NULL, recurrent_regularizer = NULL,
    bias_regularizer = NULL, activity_regularizer = NULL, kernel_constraint = NULL,
    recurrent_constraint = NULL, bias_constraint = NULL, dropout = 0,
    recurrent_dropout = 0, seed = NULL, return_sequences = FALSE,
    return_state = FALSE, go_backwards = FALSE, stateful = FALSE,
    ..., unroll = NULL)
{
    args <- capture_args2(list(filters = as_integer, kernel_size = as_integer_tuple,
        strides = as_integer_tuple, dilation_rate = as_integer_tuple,
        seed = as_integer, input_shape = normalize_shape, batch_size = as_integer,
        batch_input_shape = normalize_shape), ignore = "object")
    create_layer(keras$layers$ConvLSTM3D, object, args)
}


#' Gated Recurrent Unit - Cho et al. 2014.
#'
#' @description
#' Based on available runtime hardware and constraints, this layer
#' will choose different implementations (cuDNN-based or backend-native)
#' to maximize the performance. If a GPU is available and all
#' the arguments to the layer meet the requirement of the cuDNN kernel
#' (see below for details), the layer will use a fast cuDNN implementation
#' when using the TensorFlow backend.
#'
#' The requirements to use the cuDNN implementation are:
#'
#' 1. `activation` == `tanh`
#' 2. `recurrent_activation` == `sigmoid`
#' 3. `dropout` == 0 and `recurrent_dropout` == 0
#' 4. `unroll` is `FALSE`
#' 5. `use_bias` is `TRUE`
#' 6. `reset_after` is `TRUE`
#' 7. Inputs, if use masking, are strictly right-padded.
#' 8. Eager execution is enabled in the outermost context.
#'
#' There are two variants of the GRU implementation. The default one is based
#' on [v3](https://arxiv.org/abs/1406.1078v3) and has reset gate applied to
#' hidden state before matrix multiplication. The other one is based on
#' [original](https://arxiv.org/abs/1406.1078v1) and has the order reversed.
#'
#' The second variant is compatible with CuDNNGRU (GPU-only) and allows
#' inference on CPU. Thus it has separate biases for `kernel` and
#' `recurrent_kernel`. To use this variant, set `reset_after=TRUE` and
#' `recurrent_activation='sigmoid'`.
#'
#' For example:
#'
#' ```{r}
#' inputs <- random_uniform(c(32, 10, 8))
#' outputs <- inputs |> layer_gru(4)
#' shape(outputs)
#' # (32, 4)
#' gru <- layer_gru(, 4, return_sequences = TRUE, return_state = TRUE)
#' c(whole_sequence_output, final_state) %<-% gru(inputs)
#' shape(whole_sequence_output)
#' shape(final_state)
#' ```
#'
#' # Call Arguments
#' - `inputs`: A 3D tensor, with shape `(batch, timesteps, feature)`.
#' - `mask`: Binary tensor of shape `(samples, timesteps)` indicating whether
#'     a given timestep should be masked  (optional).
#'     An individual `TRUE` entry indicates that the corresponding timestep
#'     should be utilized, while a `FALSE` entry indicates that the
#'     corresponding timestep should be ignored. Defaults to `NULL`.
#' - `training`: Python boolean indicating whether the layer should behave in
#'     training mode or in inference mode. This argument is passed to the
#'     cell when calling it. This is only relevant if `dropout` or
#'     `recurrent_dropout` is used  (optional). Defaults to `NULL`.
#' - `initial_state`: List of initial state tensors to be passed to the first
#'     call of the cell (optional, `NULL` causes creation
#'     of zero-filled initial state tensors). Defaults to `NULL`.
#'
#' @param units
#' Positive integer, dimensionality of the output space.
#'
#' @param activation
#' Activation function to use.
#' Default: hyperbolic tangent (`tanh`).
#' If you pass `NULL`, no activation is applied
#' (ie. "linear" activation: `a(x) = x`).
#'
#' @param recurrent_activation
#' Activation function to use
#' for the recurrent step.
#' Default: sigmoid (`sigmoid`).
#' If you pass `NULL`, no activation is applied
#' (ie. "linear" activation: `a(x) = x`).
#'
#' @param use_bias
#' Boolean, (default `TRUE`), whether the layer
#' should use a bias vector.
#'
#' @param kernel_initializer
#' Initializer for the `kernel` weights matrix,
#' used for the linear transformation of the inputs. Default:
#' `"glorot_uniform"`.
#'
#' @param recurrent_initializer
#' Initializer for the `recurrent_kernel`
#' weights matrix, used for the linear transformation of the recurrent
#' state. Default: `"orthogonal"`.
#'
#' @param bias_initializer
#' Initializer for the bias vector. Default: `"zeros"`.
#'
#' @param kernel_regularizer
#' Regularizer function applied to the `kernel` weights
#' matrix. Default: `NULL`.
#'
#' @param recurrent_regularizer
#' Regularizer function applied to the
#' `recurrent_kernel` weights matrix. Default: `NULL`.
#'
#' @param bias_regularizer
#' Regularizer function applied to the bias vector.
#' Default: `NULL`.
#'
#' @param activity_regularizer
#' Regularizer function applied to the output of the
#' layer (its "activation"). Default: `NULL`.
#'
#' @param kernel_constraint
#' Constraint function applied to the `kernel` weights
#' matrix. Default: `NULL`.
#'
#' @param recurrent_constraint
#' Constraint function applied to the
#' `recurrent_kernel` weights matrix. Default: `NULL`.
#'
#' @param bias_constraint
#' Constraint function applied to the bias vector.
#' Default: `NULL`.
#'
#' @param dropout
#' Float between 0 and 1. Fraction of the units to drop for the
#' linear transformation of the inputs. Default: 0.
#'
#' @param recurrent_dropout
#' Float between 0 and 1. Fraction of the units to drop
#' for the linear transformation of the recurrent state. Default: 0.
#'
#' @param seed
#' Random seed for dropout.
#'
#' @param return_sequences
#' Boolean. Whether to return the last output
#' in the output sequence, or the full sequence. Default: `FALSE`.
#'
#' @param return_state
#' Boolean. Whether to return the last state in addition
#' to the output. Default: `FALSE`.
#'
#' @param go_backwards
#' Boolean (default `FALSE`).
#' If `TRUE`, process the input sequence backwards and return the
#' reversed sequence.
#'
#' @param stateful
#' Boolean (default: `FALSE`). If `TRUE`, the last state
#' for each sample at index i in a batch will be used as initial
#' state for the sample of index i in the following batch.
#'
#' @param unroll
#' Boolean (default: `FALSE`).
#' If `TRUE`, the network will be unrolled,
#' else a symbolic loop will be used.
#' Unrolling can speed-up a RNN,
#' although it tends to be more memory-intensive.
#' Unrolling is only suitable for short sequences.
#'
#' @param reset_after
#' GRU convention (whether to apply reset gate after or
#' before matrix multiplication). `FALSE` is `"before"`,
#' `TRUE` is `"after"` (default and cuDNN compatible).
#'
#' @param object
#' Object to compose the layer with. A tensor, array, or sequential model.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @export
#' @family gru rnn layers
#' @family rnn layers
#' @family layers
#' @seealso
#' + <https:/keras.io/api/layers/recurrent_layers/gru#gru-class>
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/layers/GRU>
#' @tether keras.layers.GRU
layer_gru <-
function (object, units, activation = "tanh", recurrent_activation = "sigmoid",
    use_bias = TRUE, kernel_initializer = "glorot_uniform", recurrent_initializer = "orthogonal",
    bias_initializer = "zeros", kernel_regularizer = NULL, recurrent_regularizer = NULL,
    bias_regularizer = NULL, activity_regularizer = NULL, kernel_constraint = NULL,
    recurrent_constraint = NULL, bias_constraint = NULL, dropout = 0,
    recurrent_dropout = 0, seed = NULL, return_sequences = FALSE,
    return_state = FALSE, go_backwards = FALSE, stateful = FALSE,
    unroll = FALSE, reset_after = TRUE, ...)
{
    args <- capture_args2(list(units = as_integer, seed = as_integer,
        input_shape = normalize_shape, batch_size = as_integer,
        batch_input_shape = normalize_shape), ignore = "object")
    create_layer(keras$layers$GRU, object, args)
}


#' Cell class for the GRU layer.
#'
#' @description
#' This class processes one step within the whole time sequence input, whereas
#' `keras.layer.GRU` processes the whole sequence.
#'
#' # Call Arguments
#' - `inputs`: A 2D tensor, with shape `(batch, features)`.
#' - `states`: A 2D tensor with shape `(batch, units)`, which is the state
#'     from the previous time step.
#' - `training`: Python boolean indicating whether the layer should behave in
#'     training mode or in inference mode. Only relevant when `dropout` or
#'     `recurrent_dropout` is used.
#'
#' # Examples
#' ```{r}
#' inputs <- random_uniform(c(32, 10, 8))
#' outputs <- inputs |> layer_rnn(rnn_cell_gru(4))
#' shape(outputs)
#' rnn <- layer_rnn(
#'    cell = rnn_cell_gru(4),
#'    return_sequences=TRUE,
#'    return_state=TRUE)
#' c(whole_sequence_output, final_state) %<-% rnn(inputs)
#' shape(whole_sequence_output)
#' shape(final_state)
#' ```
#'
#' @param units
#' Positive integer, dimensionality of the output space.
#'
#' @param activation
#' Activation function to use. Default: hyperbolic tangent
#' (`tanh`). If you pass `NULL`, no activation is applied
#' (ie. "linear" activation: `a(x) = x`).
#'
#' @param recurrent_activation
#' Activation function to use for the recurrent step.
#' Default: sigmoid (`sigmoid`). If you pass `NULL`, no activation is
#' applied (ie. "linear" activation: `a(x) = x`).
#'
#' @param use_bias
#' Boolean, (default `TRUE`), whether the layer
#' should use a bias vector.
#'
#' @param kernel_initializer
#' Initializer for the `kernel` weights matrix,
#' used for the linear transformation of the inputs. Default:
#' `"glorot_uniform"`.
#'
#' @param recurrent_initializer
#' Initializer for the `recurrent_kernel`
#' weights matrix, used for the linear transformation
#' of the recurrent state. Default: `"orthogonal"`.
#'
#' @param bias_initializer
#' Initializer for the bias vector. Default: `"zeros"`.
#'
#' @param kernel_regularizer
#' Regularizer function applied to the `kernel` weights
#' matrix. Default: `NULL`.
#'
#' @param recurrent_regularizer
#' Regularizer function applied to the
#' `recurrent_kernel` weights matrix. Default: `NULL`.
#'
#' @param bias_regularizer
#' Regularizer function applied to the bias vector.
#' Default: `NULL`.
#'
#' @param kernel_constraint
#' Constraint function applied to the `kernel` weights
#' matrix. Default: `NULL`.
#'
#' @param recurrent_constraint
#' Constraint function applied to the
#' `recurrent_kernel` weights matrix. Default: `NULL`.
#'
#' @param bias_constraint
#' Constraint function applied to the bias vector.
#' Default: `NULL`.
#'
#' @param dropout
#' Float between 0 and 1. Fraction of the units to drop for the
#' linear transformation of the inputs. Default: 0.
#'
#' @param recurrent_dropout
#' Float between 0 and 1. Fraction of the units to drop
#' for the linear transformation of the recurrent state. Default: 0.
#'
#' @param reset_after
#' GRU convention (whether to apply reset gate after or
#' before matrix multiplication). `FALSE` = `"before"`,
#' `TRUE` = `"after"` (default and cuDNN compatible).
#'
#' @param seed
#' Random seed for dropout.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @export
#' @family gru rnn layers
#' @family rnn layers
#' @family layers
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/layers/GRUCell>
#' @tether keras.layers.GRUCell
rnn_cell_gru <-
function (units, activation = "tanh", recurrent_activation = "sigmoid",
    use_bias = TRUE, kernel_initializer = "glorot_uniform", recurrent_initializer = "orthogonal",
    bias_initializer = "zeros", kernel_regularizer = NULL, recurrent_regularizer = NULL,
    bias_regularizer = NULL, kernel_constraint = NULL, recurrent_constraint = NULL,
    bias_constraint = NULL, dropout = 0, recurrent_dropout = 0,
    reset_after = TRUE, seed = NULL, ...)
{
    args <- capture_args2(list(units = as_integer, seed = as_integer,
        input_shape = normalize_shape, batch_size = as_integer,
        batch_input_shape = normalize_shape))
    do.call(keras$layers$GRUCell, args)
}


#' Long Short-Term Memory layer - Hochreiter 1997.
#'
#' @description
#' Based on available runtime hardware and constraints, this layer
#' will choose different implementations (cuDNN-based or backend-native)
#' to maximize the performance. If a GPU is available and all
#' the arguments to the layer meet the requirement of the cuDNN kernel
#' (see below for details), the layer will use a fast cuDNN implementation
#' when using the TensorFlow backend.
#' The requirements to use the cuDNN implementation are:
#'
#' 1. `activation` == `tanh`
#' 2. `recurrent_activation` == `sigmoid`
#' 3. `dropout` == 0 and `recurrent_dropout` == 0
#' 4. `unroll` is `FALSE`
#' 5. `use_bias` is `TRUE`
#' 6. Inputs, if use masking, are strictly right-padded.
#' 7. Eager execution is enabled in the outermost context.
#'
#' For example:
#'
#' ```{r}
#' input <- random_uniform(c(32, 10, 8))
#' output <- input |> layer_lstm(4)
#' shape(output)
#'
#' lstm <- layer_lstm(units = 4, return_sequences = TRUE, return_state = TRUE)
#' c(whole_seq_output, final_memory_state, final_carry_state) %<-% lstm(input)
#' shape(whole_seq_output)
#' shape(final_memory_state)
#' shape(final_carry_state)
#' ```
#'
#' # Call Arguments
#' - `inputs`: A 3D tensor, with shape `(batch, timesteps, feature)`.
#' - `mask`: Binary tensor of shape `(samples, timesteps)` indicating whether
#'     a given timestep should be masked  (optional).
#'     An individual `TRUE` entry indicates that the corresponding timestep
#'     should be utilized, while a `FALSE` entry indicates that the
#'     corresponding timestep should be ignored. Defaults to `NULL`.
#' - `training`: Boolean indicating whether the layer should behave in
#'     training mode or in inference mode. This argument is passed to the
#'     cell when calling it. This is only relevant if `dropout` or
#'     `recurrent_dropout` is used  (optional). Defaults to `NULL`.
#' - `initial_state`: List of initial state tensors to be passed to the first
#'     call of the cell (optional, `NULL` causes creation
#'     of zero-filled initial state tensors). Defaults to `NULL`.
#'
#' @param units
#' Positive integer, dimensionality of the output space.
#'
#' @param activation
#' Activation function to use.
#' Default: hyperbolic tangent (`tanh`).
#' If you pass `NULL`, no activation is applied
#' (ie. "linear" activation: `a(x) = x`).
#'
#' @param recurrent_activation
#' Activation function to use
#' for the recurrent step.
#' Default: sigmoid (`sigmoid`).
#' If you pass `NULL`, no activation is applied
#' (ie. "linear" activation: `a(x) = x`).
#'
#' @param use_bias
#' Boolean, (default `TRUE`), whether the layer
#' should use a bias vector.
#'
#' @param kernel_initializer
#' Initializer for the `kernel` weights matrix,
#' used for the linear transformation of the inputs. Default:
#' `"glorot_uniform"`.
#'
#' @param recurrent_initializer
#' Initializer for the `recurrent_kernel`
#' weights matrix, used for the linear transformation of the recurrent
#' state. Default: `"orthogonal"`.
#'
#' @param bias_initializer
#' Initializer for the bias vector. Default: `"zeros"`.
#'
#' @param unit_forget_bias
#' Boolean (default `TRUE`). If `TRUE`,
#' add 1 to the bias of the forget gate at initialization.
#' Setting it to `TRUE` will also force `bias_initializer="zeros"`.
#' This is recommended in [Jozefowicz et al.](
#' https://github.com/mlresearch/v37/blob/gh-pages/jozefowicz15.pdf)
#'
#' @param kernel_regularizer
#' Regularizer function applied to the `kernel` weights
#' matrix. Default: `NULL`.
#'
#' @param recurrent_regularizer
#' Regularizer function applied to the
#' `recurrent_kernel` weights matrix. Default: `NULL`.
#'
#' @param bias_regularizer
#' Regularizer function applied to the bias vector.
#' Default: `NULL`.
#'
#' @param activity_regularizer
#' Regularizer function applied to the output of the
#' layer (its "activation"). Default: `NULL`.
#'
#' @param kernel_constraint
#' Constraint function applied to the `kernel` weights
#' matrix. Default: `NULL`.
#'
#' @param recurrent_constraint
#' Constraint function applied to the
#' `recurrent_kernel` weights matrix. Default: `NULL`.
#'
#' @param bias_constraint
#' Constraint function applied to the bias vector.
#' Default: `NULL`.
#'
#' @param dropout
#' Float between 0 and 1. Fraction of the units to drop for the
#' linear transformation of the inputs. Default: 0.
#'
#' @param recurrent_dropout
#' Float between 0 and 1. Fraction of the units to drop
#' for the linear transformation of the recurrent state. Default: 0.
#'
#' @param seed
#' Random seed for dropout.
#'
#' @param return_sequences
#' Boolean. Whether to return the last output
#' in the output sequence, or the full sequence. Default: `FALSE`.
#'
#' @param return_state
#' Boolean. Whether to return the last state in addition
#' to the output. Default: `FALSE`.
#'
#' @param go_backwards
#' Boolean (default: `FALSE`).
#' If `TRUE`, process the input sequence backwards and return the
#' reversed sequence.
#'
#' @param stateful
#' Boolean (default: `FALSE`). If `TRUE`, the last state
#' for each sample at index i in a batch will be used as initial
#' state for the sample of index i in the following batch.
#'
#' @param unroll
#' Boolean (default `FALSE`).
#' If `TRUE`, the network will be unrolled,
#' else a symbolic loop will be used.
#' Unrolling can speed-up a RNN,
#' although it tends to be more memory-intensive.
#' Unrolling is only suitable for short sequences.
#'
#' @param object
#' Object to compose the layer with. A tensor, array, or sequential model.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @export
#' @family lstm rnn layers
#' @family rnn layers
#' @family layers
#' @seealso
#' + <https:/keras.io/api/layers/recurrent_layers/lstm#lstm-class>
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM>
#' @tether keras.layers.LSTM
layer_lstm <-
function (object, units, activation = "tanh", recurrent_activation = "sigmoid",
    use_bias = TRUE, kernel_initializer = "glorot_uniform", recurrent_initializer = "orthogonal",
    bias_initializer = "zeros", unit_forget_bias = TRUE, kernel_regularizer = NULL,
    recurrent_regularizer = NULL, bias_regularizer = NULL, activity_regularizer = NULL,
    kernel_constraint = NULL, recurrent_constraint = NULL, bias_constraint = NULL,
    dropout = 0, recurrent_dropout = 0, seed = NULL, return_sequences = FALSE,
    return_state = FALSE, go_backwards = FALSE, stateful = FALSE,
    unroll = FALSE, ...)
{
    args <- capture_args2(list(units = as_integer, seed = as_integer,
        input_shape = normalize_shape, batch_size = as_integer,
        batch_input_shape = normalize_shape), ignore = "object")
    create_layer(keras$layers$LSTM, object, args)
}


#' Cell class for the LSTM layer.
#'
#' @description
#' This class processes one step within the whole time sequence input, whereas
#' `keras.layer.LSTM` processes the whole sequence.
#'
#' # Call Arguments
#' - `inputs`: A 2D tensor, with shape `(batch, features)`.
#' - `states`: A 2D tensor with shape `(batch, units)`, which is the state
#'     from the previous time step.
#' - `training`: Boolean indicating whether the layer should behave in
#'     training mode or in inference mode. Only relevant when `dropout` or
#'     `recurrent_dropout` is used.
#'
#' # Examples
#' ```{r}
#' inputs <- random_uniform(c(32, 10, 8))
#' output <- inputs |>
#'   layer_rnn(cell = rnn_cell_lstm(4))
#' shape(output)
#'
#' rnn <- layer_rnn(cell = rnn_cell_lstm(4),
#'                  return_sequences = T,
#'                  return_state = T)
#' c(whole_sequence_output, ...final_state) %<-% rnn(inputs)
#' str(whole_sequence_output)
#' str(final_state)
#' ```
#'
#' @param units
#' Positive integer, dimensionality of the output space.
#'
#' @param activation
#' Activation function to use. Default: hyperbolic tangent
#' (`tanh`). If you pass `NULL`, no activation is applied
#' (ie. "linear" activation: `a(x) = x`).
#'
#' @param recurrent_activation
#' Activation function to use for the recurrent step.
#' Default: sigmoid (`sigmoid`). If you pass `NULL`, no activation is
#' applied (ie. "linear" activation: `a(x) = x`).
#'
#' @param use_bias
#' Boolean, (default `TRUE`), whether the layer
#' should use a bias vector.
#'
#' @param kernel_initializer
#' Initializer for the `kernel` weights matrix,
#' used for the linear transformation of the inputs. Default:
#' `"glorot_uniform"`.
#'
#' @param recurrent_initializer
#' Initializer for the `recurrent_kernel`
#' weights matrix, used for the linear transformation
#' of the recurrent state. Default: `"orthogonal"`.
#'
#' @param bias_initializer
#' Initializer for the bias vector. Default: `"zeros"`.
#'
#' @param unit_forget_bias
#' Boolean (default `TRUE`). If `TRUE`,
#' add 1 to the bias of the forget gate at initialization.
#' Setting it to `TRUE` will also force `bias_initializer="zeros"`.
#' This is recommended in [Jozefowicz et al.](
#' https://github.com/mlresearch/v37/blob/gh-pages/jozefowicz15.pdf)
#'
#' @param kernel_regularizer
#' Regularizer function applied to the `kernel` weights
#' matrix. Default: `NULL`.
#'
#' @param recurrent_regularizer
#' Regularizer function applied to the
#' `recurrent_kernel` weights matrix. Default: `NULL`.
#'
#' @param bias_regularizer
#' Regularizer function applied to the bias vector.
#' Default: `NULL`.
#'
#' @param kernel_constraint
#' Constraint function applied to the `kernel` weights
#' matrix. Default: `NULL`.
#'
#' @param recurrent_constraint
#' Constraint function applied to the
#' `recurrent_kernel` weights matrix. Default: `NULL`.
#'
#' @param bias_constraint
#' Constraint function applied to the bias vector.
#' Default: `NULL`.
#'
#' @param dropout
#' Float between 0 and 1. Fraction of the units to drop for the
#' linear transformation of the inputs. Default: 0.
#'
#' @param recurrent_dropout
#' Float between 0 and 1. Fraction of the units to drop
#' for the linear transformation of the recurrent state. Default: 0.
#'
#' @param seed
#' Random seed for dropout.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @export
#' @family lstm rnn layers
#' @family rnn layers
#' @family layers
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTMCell>
#' @tether keras.layers.LSTMCell
rnn_cell_lstm <-
function (units, activation = "tanh", recurrent_activation = "sigmoid",
    use_bias = TRUE, kernel_initializer = "glorot_uniform", recurrent_initializer = "orthogonal",
    bias_initializer = "zeros", unit_forget_bias = TRUE, kernel_regularizer = NULL,
    recurrent_regularizer = NULL, bias_regularizer = NULL, kernel_constraint = NULL,
    recurrent_constraint = NULL, bias_constraint = NULL, dropout = 0,
    recurrent_dropout = 0, seed = NULL, ...)
{
    args <- capture_args2(list(units = as_integer, seed = as_integer,
        input_shape = normalize_shape, batch_size = as_integer,
        batch_input_shape = normalize_shape))
    do.call(keras$layers$LSTMCell, args)
}


#' Base class for recurrent layers
#'
#' @description
#'
#' # Call Arguments
#' - `inputs`: Input tensor.
#' - `initial_state`: List of initial state tensors to be passed to the first
#'     call of the cell.
#' - `mask`: Binary tensor of shape `[batch_size, timesteps]`
#'     indicating whether a given timestep should be masked.
#'     An individual `TRUE` entry indicates that the corresponding
#'     timestep should be utilized, while a `FALSE` entry indicates
#'     that the corresponding timestep should be ignored.
#' - `training`: Python boolean indicating whether the layer should behave in
#'     training mode or in inference mode. This argument is passed
#'     to the cell when calling it.
#'     This is for use with cells that use dropout.
#'
#' # Input Shape
#' 3-D tensor with shape `(batch_size, timesteps, features)`.
#'
#' # Output Shape
#' - If `return_state`: a list of tensors. The first tensor is
#' the output. The remaining tensors are the last states,
#' each with shape `(batch_size, state_size)`, where `state_size` could
#' be a high dimension tensor shape.
#' - If `return_sequences`: 3D tensor with shape
#' `(batch_size, timesteps, output_size)`.
#'
#' # Masking:
#'
#' This layer supports masking for input data with a variable number
#' of timesteps. To introduce masks to your data,
#' use a `layer_Embedding` layer with the `mask_zero` parameter
#' set to `TRUE`.
#'
#' Note on using statefulness in RNNs:
#'
#' You can set RNN layers to be 'stateful', which means that the states
#' computed for the samples in one batch will be reused as initial states
#' for the samples in the next batch. This assumes a one-to-one mapping
#' between samples in different successive batches.
#'
#' To enable statefulness:
#'
#' - Specify `stateful=TRUE` in the layer constructor.
#' - Specify a fixed batch size for your model, by passing
#' If sequential model:
#'     `batch_input_shape=(...)` to the first layer in your model.
#' Else for functional model with 1 or more Input layers:
#'     `batch_shape=(...)` to all the first layers in your model.
#' This is the expected shape of your inputs
#' *including the batch size*.
#' It should be a list of integers, e.g. `(32, 10, 100)`.
#' - Specify `shuffle=FALSE` when calling `fit()`.
#'
#' To reset the states of your model, call [`reset_state()`] on either
#' a specific layer, or on your entire model.
#'
#' Note on specifying the initial state of RNNs:
#'
#' You can specify the initial state of RNN layers symbolically by
#' calling them with the keyword argument `initial_state`. The value of
#' `initial_state` should be a tensor or list of tensors representing
#' the initial state of the RNN layer.
#'
#'
#' # Examples
#' ```{r}
#' # First, let's define a RNN Cell, as a layer subclass.
#' MinimalRNNCell(keras$layers$Layer) %py_class% {
#'   initialize <- function(units, ...) {
#'     super$initialize(...)
#'     self$units <- as.integer(units)
#'     self$state_size <- as.integer(units)
#'   }
#'
#'   build <- function(input_shape) {
#'     self$kernel <- self$add_weight(
#'       shape = c(tail(input_shape, 1), self$units),
#'       initializer = 'uniform',
#'       name = 'kernel')
#'     self$recurrent_kernel <- self$add_weight(
#'       shape = c(self$units, self$units),
#'       initializer = 'uniform',
#'       name = 'recurrent_kernel')
#'     self$built <- TRUE
#'   }
#'
#'   call <- function(inputs, states) {
#'     prev_output <- states[[1]]
#'     h <- op_matmul(inputs, self$kernel)
#'     output <- h + op_matmul(prev_output, self$recurrent_kernel)
#'     list(output, list(output))
#'   }
#'
#' }
#'
#' # Let's use this cell in a RNN layer:
#'
#' cell <- MinimalRNNCell(units = 32)
#' x <- layer_input(shape = shape(NULL, 5))
#' layer <- layer_rnn(cell = cell)
#' y <- layer(x)
#'
#' # Here's how to use the cell to build a stacked RNN:
#'
#' cells <- list(MinimalRNNCell(units = 32), MinimalRNNCell(units = 4))
#' x <- layer_input(shape = shape(NULL, 5))
#' layer <- layer_rnn(cell = cells)
#' y <- layer(x)
#' ```
#'
#' @param cell
#' A RNN cell instance or a list of RNN cell instances.
#' A RNN cell is a class that has:
#' - A `call(input_at_t, states_at_t)` method, returning
#' `(output_at_t, states_at_t_plus_1)`. The call method of the
#' cell can also take the optional argument `constants`, see
#' section "Note on passing external constants" below.
#' - A `state_size` attribute. This can be a single integer
#' (single state) in which case it is the size of the recurrent
#' state. This can also be a list of integers
#' (one size per state).
#' - A `output_size` attribute, a single integer.
#' - A `get_initial_state(batch_size=NULL)`
#' method that creates a tensor meant to be fed to `call()` as the
#' initial state, if the user didn't specify any initial state
#' via other means. The returned initial state should have
#' shape `(batch_size, cell.state_size)`.
#' The cell might choose to create a tensor full of zeros,
#' or other values based on the cell's implementation.
#' `inputs` is the input tensor to the RNN layer, with shape
#' `(batch_size, timesteps, features)`.
#' If this method is not implemented
#' by the cell, the RNN layer will create a zero filled tensor
#' with shape `(batch_size, cell$state_size)`.
#' In the case that `cell` is a list of RNN cell instances, the cells
#' will be stacked on top of each other in the RNN, resulting in an
#' efficient stacked RNN.
#'
#' @param return_sequences
#' Boolean (default `FALSE`). Whether to return the last
#' output in the output sequence, or the full sequence.
#'
#' @param return_state
#' Boolean (default `FALSE`).
#' Whether to return the last state in addition to the output.
#'
#' @param go_backwards
#' Boolean (default `FALSE`).
#' If `TRUE`, process the input sequence backwards and return the
#' reversed sequence.
#'
#' @param stateful
#' Boolean (default `FALSE`). If TRUE, the last state
#' for each sample at index `i` in a batch will be used as initial
#' state for the sample of index `i` in the following batch.
#'
#' @param unroll
#' Boolean (default `FALSE`).
#' If TRUE, the network will be unrolled, else a symbolic loop will be
#' used. Unrolling can speed-up a RNN, although it tends to be more
#' memory-intensive. Unrolling is only suitable for short sequences.
#'
#' @param zero_output_for_mask
#' Boolean (default `FALSE`).
#' Whether the output should use zeros for the masked timesteps.
#' Note that this field is only used when `return_sequences`
#' is `TRUE` and `mask` is provided.
#' It can useful if you want to reuse the raw output sequence of
#' the RNN without interference from the masked timesteps, e.g.,
#' merging bidirectional RNNs.
#'
#' @param object
#' Object to compose the layer with. A tensor, array, or sequential model.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @export
#' @family rnn layers
#' @family layers
#' @seealso
#' + <https:/keras.io/api/layers/recurrent_layers/rnn#rnn-class>
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/layers/RNN>
#'
#' @tether keras.layers.RNN
layer_rnn <-
function (object, cell, return_sequences = FALSE, return_state = FALSE,
    go_backwards = FALSE, stateful = FALSE, unroll = FALSE, zero_output_for_mask = FALSE,
    ...)
{
    args <- capture_args2(list(cell = as_integer, input_shape = normalize_shape,
        batch_size = as_integer, batch_input_shape = normalize_shape),
        ignore = "object")
    create_layer(keras$layers$RNN, object, args)
}


#' Fully-connected RNN where the output is to be fed back as the new input.
#'
#' @description
#'
#' # Call Arguments
#' - `sequence`: A 3D tensor, with shape `[batch, timesteps, feature]`.
#' - `mask`: Binary tensor of shape `[batch, timesteps]` indicating whether
#'     a given timestep should be masked. An individual `TRUE` entry
#'     indicates that the corresponding timestep should be utilized,
#'     while a `FALSE` entry indicates that the corresponding timestep
#'     should be ignored.
#' - `training`: Python boolean indicating whether the layer should behave in
#'     training mode or in inference mode.
#'     This argument is passed to the cell when calling it.
#'     This is only relevant if `dropout` or `recurrent_dropout` is used.
#' - `initial_state`: List of initial state tensors to be passed to the first
#'     call of the cell.
#'
#' # Examples
#' ```{r}
#' inputs <- random_uniform(c(32, 10, 8))
#' simple_rnn <- layer_simple_rnn(units = 4)
#' output <- simple_rnn(inputs)  # The output has shape `(32, 4)`.
#' simple_rnn <- layer_simple_rnn(
#'     units = 4, return_sequences=TRUE, return_state=TRUE
#' )
#' # whole_sequence_output has shape `(32, 10, 4)`.
#' # final_state has shape `(32, 4)`.
#' c(whole_sequence_output, final_state) %<-% simple_rnn(inputs)
#' ```
#'
#' @param units
#' Positive integer, dimensionality of the output space.
#'
#' @param activation
#' Activation function to use.
#' Default: hyperbolic tangent (`tanh`).
#' If you pass NULL, no activation is applied
#' (ie. "linear" activation: `a(x) = x`).
#'
#' @param use_bias
#' Boolean, (default `TRUE`), whether the layer uses
#' a bias vector.
#'
#' @param kernel_initializer
#' Initializer for the `kernel` weights matrix,
#' used for the linear transformation of the inputs. Default:
#' `"glorot_uniform"`.
#'
#' @param recurrent_initializer
#' Initializer for the `recurrent_kernel`
#' weights matrix, used for the linear transformation of the recurrent
#' state.  Default: `"orthogonal"`.
#'
#' @param bias_initializer
#' Initializer for the bias vector. Default: `"zeros"`.
#'
#' @param kernel_regularizer
#' Regularizer function applied to the `kernel` weights
#' matrix. Default: `NULL`.
#'
#' @param recurrent_regularizer
#' Regularizer function applied to the
#' `recurrent_kernel` weights matrix. Default: `NULL`.
#'
#' @param bias_regularizer
#' Regularizer function applied to the bias vector.
#' Default: `NULL`.
#'
#' @param activity_regularizer
#' Regularizer function applied to the output of the
#' layer (its "activation"). Default: `NULL`.
#'
#' @param kernel_constraint
#' Constraint function applied to the `kernel` weights
#' matrix. Default: `NULL`.
#'
#' @param recurrent_constraint
#' Constraint function applied to the
#' `recurrent_kernel` weights matrix.  Default: `NULL`.
#'
#' @param bias_constraint
#' Constraint function applied to the bias vector.
#' Default: `NULL`.
#'
#' @param dropout
#' Float between 0 and 1.
#' Fraction of the units to drop for the linear transformation
#' of the inputs. Default: 0.
#'
#' @param recurrent_dropout
#' Float between 0 and 1.
#' Fraction of the units to drop for the linear transformation of the
#' recurrent state. Default: 0.
#'
#' @param return_sequences
#' Boolean. Whether to return the last output
#' in the output sequence, or the full sequence. Default: `FALSE`.
#'
#' @param return_state
#' Boolean. Whether to return the last state
#' in addition to the output. Default: `FALSE`.
#'
#' @param go_backwards
#' Boolean (default: `FALSE`).
#' If `TRUE`, process the input sequence backwards and return the
#' reversed sequence.
#'
#' @param stateful
#' Boolean (default: `FALSE`). If `TRUE`, the last state
#' for each sample at index i in a batch will be used as initial
#' state for the sample of index i in the following batch.
#'
#' @param unroll
#' Boolean (default: `FALSE`).
#' If `TRUE`, the network will be unrolled,
#' else a symbolic loop will be used.
#' Unrolling can speed-up a RNN,
#' although it tends to be more memory-intensive.
#' Unrolling is only suitable for short sequences.
#'
#' @param object
#' Object to compose the layer with. A tensor, array, or sequential model.
#'
#' @param seed
#' Initial seed for the random number generator
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @export
#' @family simple rnn layers
#' @family rnn layers
#' @family layers
#' @seealso
#' + <https:/keras.io/api/layers/recurrent_layers/simple_rnn#simplernn-class>
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/layers/SimpleRNN>
#'
#' @tether keras.layers.SimpleRNN
layer_simple_rnn <-
function (object, units, activation = "tanh", use_bias = TRUE,
    kernel_initializer = "glorot_uniform", recurrent_initializer = "orthogonal",
    bias_initializer = "zeros", kernel_regularizer = NULL, recurrent_regularizer = NULL,
    bias_regularizer = NULL, activity_regularizer = NULL, kernel_constraint = NULL,
    recurrent_constraint = NULL, bias_constraint = NULL, dropout = 0,
    recurrent_dropout = 0, return_sequences = FALSE, return_state = FALSE,
    go_backwards = FALSE, stateful = FALSE, unroll = FALSE, seed = NULL,
    ...)
{
    args <- capture_args2(list(units = as_integer, seed = as_integer,
        input_shape = normalize_shape, batch_size = as_integer,
        batch_input_shape = normalize_shape), ignore = "object")
    create_layer(keras$layers$SimpleRNN, object, args)
}


#' Cell class for SimpleRNN.
#'
#' @description
#' This class processes one step within the whole time sequence input, whereas
#' `keras.layer.SimpleRNN` processes the whole sequence.
#'
#' # Call Arguments
#' - `sequence`: A 2D tensor, with shape `(batch, features)`.
#' - `states`: A 2D tensor with shape `(batch, units)`, which is the state
#'     from the previous time step.
#' - `training`: Python boolean indicating whether the layer should behave in
#'     training mode or in inference mode. Only relevant when `dropout` or
#'     `recurrent_dropout` is used.
#'
#' # Examples
#' ```{r}
#' inputs <- random_uniform(c(32, 10, 8))
#' rnn <- layer_rnn(cell = rnn_cell_simple(units = 4))
#' output <- rnn(inputs)  # The output has shape `(32, 4)`.
#' rnn <- layer_rnn(
#'     cell = rnn_cell_simple(units = 4),
#'     return_sequences=TRUE,
#'     return_state=TRUE
#' )
#' # whole_sequence_output has shape `(32, 10, 4)`.
#' # final_state has shape `(32, 4)`.
#' c(whole_sequence_output, final_state) %<-% rnn(inputs)
#' ```
#'
#' @param units
#' Positive integer, dimensionality of the output space.
#'
#' @param activation
#' Activation function to use.
#' Default: hyperbolic tangent (`tanh`).
#' If you pass `NULL`, no activation is applied
#' (ie. "linear" activation: `a(x) = x`).
#'
#' @param use_bias
#' Boolean, (default `TRUE`), whether the layer
#' should use a bias vector.
#'
#' @param kernel_initializer
#' Initializer for the `kernel` weights matrix,
#' used for the linear transformation of the inputs. Default:
#' `"glorot_uniform"`.
#'
#' @param recurrent_initializer
#' Initializer for the `recurrent_kernel`
#' weights matrix, used for the linear transformation
#' of the recurrent state. Default: `"orthogonal"`.
#'
#' @param bias_initializer
#' Initializer for the bias vector. Default: `"zeros"`.
#'
#' @param kernel_regularizer
#' Regularizer function applied to the `kernel` weights
#' matrix. Default: `NULL`.
#'
#' @param recurrent_regularizer
#' Regularizer function applied to the
#' `recurrent_kernel` weights matrix. Default: `NULL`.
#'
#' @param bias_regularizer
#' Regularizer function applied to the bias vector.
#' Default: `NULL`.
#'
#' @param kernel_constraint
#' Constraint function applied to the `kernel` weights
#' matrix. Default: `NULL`.
#'
#' @param recurrent_constraint
#' Constraint function applied to the
#' `recurrent_kernel` weights matrix. Default: `NULL`.
#'
#' @param bias_constraint
#' Constraint function applied to the bias vector.
#' Default: `NULL`.
#'
#' @param dropout
#' Float between 0 and 1. Fraction of the units to drop for the
#' linear transformation of the inputs. Default: 0.
#'
#' @param recurrent_dropout
#' Float between 0 and 1. Fraction of the units to drop
#' for the linear transformation of the recurrent state. Default: 0.
#'
#' @param seed
#' Random seed for dropout.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @export
#' @family simple rnn layers
#' @family rnn layers
#' @family layers
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/layers/SimpleRNNCell>
#'
#' @tether keras.layers.SimpleRNNCell
rnn_cell_simple <-
function (units, activation = "tanh", use_bias = TRUE, kernel_initializer = "glorot_uniform",
    recurrent_initializer = "orthogonal", bias_initializer = "zeros",
    kernel_regularizer = NULL, recurrent_regularizer = NULL,
    bias_regularizer = NULL, kernel_constraint = NULL, recurrent_constraint = NULL,
    bias_constraint = NULL, dropout = 0, recurrent_dropout = 0,
    seed = NULL, ...)
{
    args <- capture_args2(list(units = as_integer, seed = as_integer,
        input_shape = normalize_shape, batch_size = as_integer,
        batch_input_shape = normalize_shape))
    do.call(keras$layers$SimpleRNNCell, args)
}


#' Wrapper allowing a stack of RNN cells to behave as a single cell.
#'
#' @description
#' Used to implement efficient stacked RNNs.
#'
#' # Examples
#' ```{r}
#' batch_size <- 3
#' sentence_length <- 5
#' num_features <- 2
#' new_shape <- c(batch_size, sentence_length, num_features)
#' x <- array(1:30, dim = new_shape)
#'
#' rnn_cells <- lapply(1:2, function(x) rnn_cell_lstm(units = 128))
#' stacked_lstm <- rnn_cells_stack(rnn_cells)
#' lstm_layer <- layer_rnn(cell = stacked_lstm)
#'
#' result <- lstm_layer(x)
#' ```
#'
#' @param cells
#' List of RNN cell instances.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @export
#' @family rnn layers
#' @family layers
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/layers/StackedRNNCells>
#'
#' @tether keras.layers.StackedRNNCells
rnn_cells_stack <-
function (cells, ...)
{
    args <- capture_args2(list(input_shape = normalize_shape,
        batch_size = as_integer, batch_input_shape = normalize_shape))
    do.call(keras$layers$StackedRNNCells, args)
}


#' This wrapper allows to apply a layer to every temporal slice of an input.
#'
#' @description
#' Every input should be at least 3D, and the dimension of index one of the
#' first input will be considered to be the temporal dimension.
#'
#' Consider a batch of 32 video samples, where each sample is a 128x128 RGB
#' image with `channels_last` data format, across 10 timesteps.
#' The batch input shape is `(32, 10, 128, 128, 3)`.
#'
#' You can then use `TimeDistributed` to apply the same `Conv2D` layer to each
#' of the 10 timesteps, independently:
#'
#' ```{r}
#' inputs <- layer_input(shape = c(10, 128, 128, 3), batch_size = 32)
#' conv_2d_layer <- layer_conv_2d(filters = 64, kernel_size = c(3, 3))
#' outputs <- layer_time_distributed(inputs, layer = conv_2d_layer)
#' shape(outputs)
#' ```
#'
#' Because `layer_time_distributed` applies the same instance of `layer_conv2d` to each of
#' the timestamps, the same set of weights are used at each timestamp.
#'
#' # Call Arguments
#' - `inputs`: Input tensor of shape (batch, time, ...) or nested tensors,
#'     and each of which has shape (batch, time, ...).
#' - `training`: Boolean indicating whether the layer should behave in
#'     training mode or in inference mode. This argument is passed to the
#'     wrapped layer (only if the layer supports this argument).
#' - `mask`: Binary tensor of shape `(samples, timesteps)` indicating whether
#'     a given timestep should be masked. This argument is passed to the
#'     wrapped layer (only if the layer supports this argument).
#'
#' @param layer
#' a `layer_Layer` instance.
#'
#' @param object
#' Object to compose the layer with. A tensor, array, or sequential model.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @export
#' @family rnn layers
#' @family layers
#' @seealso
#' + <https:/keras.io/api/layers/recurrent_layers/time_distributed#timedistributed-class>
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/layers/TimeDistributed>
#'
#' @tether keras.layers.TimeDistributed
layer_time_distributed <-
function (object, layer, ...)
{
    args <- capture_args2(list(input_shape = normalize_shape,
        batch_size = as_integer, batch_input_shape = normalize_shape),
        ignore = "object")
    create_layer(keras$layers$TimeDistributed, object, args)
}
