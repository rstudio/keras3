layer_upsampling_1d <-
function (object, size = 2L, ...) 
{
    args <- capture_args2(list(size = as_integer, input_shape = normalize_shape, 
        batch_size = as_integer, batch_input_shape = normalize_shape), 
        ignore = "object")
    create_layer(keras$layers$UpSampling1D, object, args)
}
