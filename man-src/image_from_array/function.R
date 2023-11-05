image_from_array <-
function (x, data_format = NULL, scale = TRUE, dtype = NULL) 
{
    args <- capture_args2(NULL)
    do.call(keras$utils$array_to_img, args)
}
