layer_dense <-
function (object, units, activation = NULL, use_bias = TRUE, 
    kernel_initializer = "glorot_uniform", bias_initializer = "zeros", 
    kernel_regularizer = NULL, bias_regularizer = NULL, activity_regularizer = NULL, 
    kernel_constraint = NULL, bias_constraint = NULL, ...) 
{
    args <- capture_args2(list(units = as_integer, input_shape = normalize_shape, 
        batch_size = as_integer, batch_input_shape = normalize_shape), 
        ignore = "object")
    create_layer(keras$layers$Dense, object, args)
}
