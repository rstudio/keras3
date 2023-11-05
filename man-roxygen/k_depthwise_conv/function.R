k_depthwise_conv <-
function (inputs, kernel, strides = 1L, padding = "valid", data_format = NULL, 
    dilation_rate = 1L) 
{
    args <- capture_args2(list(strides = as_integer, dilation_rate = as_integer))
    do.call(keras$ops$depthwise_conv, args)
}
