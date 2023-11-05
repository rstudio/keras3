layer_cropping_2d <-
function (object, cropping = list(list(0L, 0L), list(0L, 0L)), 
    data_format = NULL, ...) 
{
    args <- capture_args2(list(cropping = as_integer, padding = function (x) 
    normalize_cropping(x, 2L), input_shape = normalize_shape, 
        batch_size = as_integer, batch_input_shape = normalize_shape), 
        ignore = "object")
    create_layer(keras$layers$Cropping2D, object, args)
}
