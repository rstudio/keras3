layer_depthwise_conv_2d <-
function (object, kernel_size, strides = list(1L, 1L), padding = "valid", 
    depth_multiplier = 1L, data_format = NULL, dilation_rate = list(
        1L, 1L), activation = NULL, use_bias = TRUE, depthwise_initializer = "glorot_uniform", 
    bias_initializer = "zeros", depthwise_regularizer = NULL, 
    bias_regularizer = NULL, activity_regularizer = NULL, depthwise_constraint = NULL, 
    bias_constraint = NULL, ...) 
{
    args <- capture_args2(list(kernel_size = as_integer, strides = as_integer, 
        depth_multiplier = as_integer, dilation_rate = as_integer, 
        input_shape = normalize_shape, batch_size = as_integer, 
        batch_input_shape = normalize_shape), ignore = "object")
    create_layer(keras$layers$DepthwiseConv2D, object, args)
}
