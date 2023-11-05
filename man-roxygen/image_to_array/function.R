image_to_array <-
function (img, data_format = NULL, dtype = NULL) 
{
    args <- capture_args2(NULL)
    do.call(keras$utils$img_to_array, args)
}
