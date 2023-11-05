config_image_data_format <-
function () 
{
    args <- capture_args2(NULL)
    do.call(keras$config$image_data_format, args)
}
