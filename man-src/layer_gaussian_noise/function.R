layer_gaussian_noise <-
function (object, stddev, seed = NULL, ...) 
{
    args <- capture_args2(list(seed = as_integer, input_shape = normalize_shape, 
        batch_size = as_integer, batch_input_shape = normalize_shape), 
        ignore = "object")
    create_layer(keras$layers$GaussianNoise, object, args)
}
