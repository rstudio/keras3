layer_random_rotation <-
function (object, factor, fill_mode = "reflect", interpolation = "bilinear", 
    seed = NULL, fill_value = 0, value_range = list(0L, 255L), 
    data_format = NULL, ...) 
{
    args <- capture_args2(list(seed = as_integer, input_shape = normalize_shape, 
        batch_size = as_integer, batch_input_shape = normalize_shape), 
        ignore = "object")
    create_layer(keras$layers$RandomRotation, object, args)
}
