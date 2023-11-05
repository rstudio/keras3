layer_einsum_dense <-
function (object, equation, output_shape, activation = NULL, 
    bias_axes = NULL, kernel_initializer = "glorot_uniform", 
    bias_initializer = "zeros", kernel_regularizer = NULL, bias_regularizer = NULL, 
    kernel_constraint = NULL, bias_constraint = NULL, ...) 
{
    args <- capture_args2(list(input_shape = normalize_shape, 
        batch_size = as_integer, batch_input_shape = normalize_shape), 
        ignore = "object")
    create_layer(keras$layers$EinsumDense, object, args)
}
