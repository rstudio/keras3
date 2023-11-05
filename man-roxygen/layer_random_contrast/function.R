layer_random_contrast <-
function (object, factor, seed = NULL, ...) 
{
    args <- capture_args2(list(seed = as_integer, input_shape = normalize_shape, 
        batch_size = as_integer, batch_input_shape = normalize_shape), 
        ignore = "object")
    create_layer(keras$layers$RandomContrast, object, args)
}
