layer_cropping_3d <-
function (object, cropping = list(list(1L, 1L), list(1L, 1L), 
    list(1L, 1L)), data_format = NULL, ...) 
{
    args <- capture_args2(list(cropping = as_integer, padding = function (x) 
    normalize_cropping(x, 3L), input_shape = normalize_shape, 
        batch_size = as_integer, batch_input_shape = normalize_shape), 
        ignore = "object")
    create_layer(keras$layers$Cropping3D, object, args)
}
