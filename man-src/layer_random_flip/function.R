layer_random_flip <-
function (object, mode = "horizontal_and_vertical", seed = NULL, 
    ...) 
{
    args <- capture_args2(list(seed = as_integer, input_shape = normalize_shape, 
        batch_size = as_integer, batch_input_shape = normalize_shape), 
        ignore = "object")
    create_layer(keras$layers$RandomFlip, object, args)
}
