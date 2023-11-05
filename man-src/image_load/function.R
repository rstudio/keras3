image_load <-
function (path, color_mode = "rgb", target_size = NULL, interpolation = "nearest", 
    keep_aspect_ratio = FALSE) 
{
    args <- capture_args2(NULL)
    do.call(keras$utils$load_img, args)
}
