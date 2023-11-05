layer_upsampling_2d <-
function (object, size = list(2L, 2L), data_format = NULL, interpolation = "nearest", 
    ...) 
{
    args <- capture_args2(list(size = as_integer, input_shape = normalize_shape, 
        batch_size = as_integer, batch_input_shape = normalize_shape), 
        ignore = "object")
    create_layer(keras$layers$UpSampling2D, object, args)
}
