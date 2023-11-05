layer_simple_rnn_cell <-
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
