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
