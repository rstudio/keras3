k_conv_transpose <-
function (inputs, kernel, strides, padding = "valid", output_padding = NULL, 
    data_format = NULL, dilation_rate = 1L) 
{
    args <- capture_args2(list(strides = as_integer, output_padding = as_integer, 
        dilation_rate = as_integer))
    do.call(keras$ops$conv_transpose, args)
}
