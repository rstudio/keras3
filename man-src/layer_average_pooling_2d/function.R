layer_average_pooling_2d <-
function (object, pool_size, strides = NULL, padding = "valid", 
    data_format = "channels_last", name = NULL, ...) 
{
    args <- capture_args2(list(pool_size = as_integer, strides = as_integer, 
        input_shape = normalize_shape, batch_size = as_integer, 
        batch_input_shape = normalize_shape), ignore = "object")
    create_layer(keras$layers$AveragePooling2D, object, args)
}
