layer_conv_2d_transpose <-
function (object, filters, kernel_size, strides = list(1L, 1L), 
    padding = "valid", data_format = NULL, dilation_rate = list(
        1L, 1L), activation = NULL, use_bias = TRUE, kernel_initializer = "glorot_uniform", 
    bias_initializer = "zeros", kernel_regularizer = NULL, bias_regularizer = NULL, 
    activity_regularizer = NULL, kernel_constraint = NULL, bias_constraint = NULL, 
    ...) 
{
    args <- capture_args2(list(filters = as_integer, kernel_size = as_integer_tuple, 
        strides = as_integer_tuple, dilation_rate = as_integer_tuple, 
        input_shape = normalize_shape, batch_size = as_integer, 
        batch_input_shape = normalize_shape), ignore = "object")
    create_layer(keras$layers$Conv2DTranspose, object, args)
}
