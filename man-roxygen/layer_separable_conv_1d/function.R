layer_separable_conv_1d <-
function (object, filters, kernel_size, strides = 1L, padding = "valid", 
    data_format = NULL, dilation_rate = 1L, depth_multiplier = 1L, 
    activation = NULL, use_bias = TRUE, depthwise_initializer = "glorot_uniform", 
    pointwise_initializer = "glorot_uniform", bias_initializer = "zeros", 
    depthwise_regularizer = NULL, pointwise_regularizer = NULL, 
    bias_regularizer = NULL, activity_regularizer = NULL, depthwise_constraint = NULL, 
    pointwise_constraint = NULL, bias_constraint = NULL, ...) 
{
    args <- capture_args2(list(filters = as_integer, kernel_size = as_integer, 
        strides = as_integer, dilation_rate = as_integer, depth_multiplier = as_integer, 
        input_shape = normalize_shape, batch_size = as_integer, 
        batch_input_shape = normalize_shape), ignore = "object")
    create_layer(keras$layers$SeparableConv1D, object, args)
}
