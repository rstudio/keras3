layer_spatial_dropout_1d <-
function (object, rate, seed = NULL, name = NULL, dtype = NULL) 
{
    args <- capture_args2(list(seed = as_integer, input_shape = normalize_shape, 
        batch_size = as_integer, batch_input_shape = normalize_shape), 
        ignore = "object")
    create_layer(keras$layers$SpatialDropout1D, object, args)
}
