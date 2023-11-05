k_image_extract_patches <-
function (image, size, strides = NULL, dilation_rate = 1L, padding = "valid", 
    data_format = "channels_last") 
{
    args <- capture_args2(list(size = as_integer, dilation_rate = as_integer))
    do.call(keras$ops$image$extract_patches, args)
}
