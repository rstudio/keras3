config_set_image_data_format <-
function (data_format) 
{
    args <- capture_args2(NULL)
    do.call(keras$config$set_image_data_format, args)
}
