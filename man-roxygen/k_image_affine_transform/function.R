k_image_affine_transform <-
function (image, transform, interpolation = "bilinear", fill_mode = "constant", 
    fill_value = 0L, data_format = "channels_last") 
{
    args <- capture_args2(list(fill_value = as_integer))
    do.call(keras$ops$image$affine_transform, args)
}
