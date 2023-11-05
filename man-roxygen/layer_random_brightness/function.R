layer_random_brightness <-
function (object, factor, value_range = list(0L, 255L), seed = NULL, 
    ...) 
{
    args <- capture_args2(list(seed = as_integer, input_shape = normalize_shape, 
        batch_size = as_integer, batch_input_shape = normalize_shape), 
        ignore = "object")
    create_layer(keras$layers$RandomBrightness, object, args)
}
