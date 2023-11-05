layer_global_average_pooling_3d <-
function (object, data_format = NULL, keepdims = FALSE, ...) 
{
    args <- capture_args2(list(input_shape = normalize_shape, 
        batch_size = as_integer, batch_input_shape = normalize_shape), 
        ignore = "object")
    create_layer(keras$layers$GlobalAveragePooling3D, object, 
        args)
}
