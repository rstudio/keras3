layer_spatial_dropout_2d <-
function (object, rate, data_format = NULL, seed = NULL, name = NULL, 
    dtype = NULL) 
{
    args <- capture_args2(list(seed = as_integer, input_shape = normalize_shape, 
        batch_size = as_integer, batch_input_shape = normalize_shape), 
        ignore = "object")
    create_layer(keras$layers$SpatialDropout2D, object, args)
}
