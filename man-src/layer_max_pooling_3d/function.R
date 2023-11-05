layer_max_pooling_3d <-
function (object, pool_size = list(2L, 2L, 2L), strides = NULL, 
    padding = "valid", data_format = NULL, name = NULL, ...) 
{
    args <- capture_args2(list(pool_size = as_integer, strides = as_integer, 
        input_shape = normalize_shape, batch_size = as_integer, 
        batch_input_shape = normalize_shape), ignore = "object")
    create_layer(keras$layers$MaxPooling3D, object, args)
}
