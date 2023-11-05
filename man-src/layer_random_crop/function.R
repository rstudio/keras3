layer_random_crop <-
function (object, height, width, seed = NULL, data_format = NULL, 
    name = NULL, ...) 
{
    args <- capture_args2(list(height = as_integer, width = as_integer, 
        seed = as_integer, input_shape = normalize_shape, batch_size = as_integer, 
        batch_input_shape = normalize_shape), ignore = "object")
    create_layer(keras$layers$RandomCrop, object, args)
}
