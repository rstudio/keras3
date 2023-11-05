image_array_save <-
function (x, path, data_format = NULL, file_format = NULL, scale = TRUE, 
    ...) 
{
    args <- capture_args2(NULL)
    do.call(keras$utils$save_img, args)
}
