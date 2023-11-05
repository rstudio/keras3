layer_normalization <-
function (object, axis = -1L, mean = NULL, variance = NULL, invert = FALSE, 
    ...) 
{
    args <- capture_args2(list(axis = as_axis, input_shape = normalize_shape, 
        batch_size = as_integer, batch_input_shape = normalize_shape), 
        ignore = "object")
    create_layer(keras$layers$Normalization, object, args)
}
