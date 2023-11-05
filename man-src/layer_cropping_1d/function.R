layer_cropping_1d <-
function (object, cropping = list(1L, 1L), ...) 
{
    args <- capture_args2(list(cropping = as_integer, input_shape = normalize_shape, 
        batch_size = as_integer, batch_input_shape = normalize_shape), 
        ignore = "object")
    create_layer(keras$layers$Cropping1D, object, args)
}
