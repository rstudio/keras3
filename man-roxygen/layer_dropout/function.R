layer_dropout <-
function (object, rate, noise_shape = NULL, seed = NULL, ...) 
{
    args <- capture_args2(list(noise_shape = as_integer, seed = as_integer, 
        input_shape = normalize_shape, batch_size = as_integer, 
        batch_input_shape = normalize_shape), ignore = "object")
    create_layer(keras$layers$Dropout, object, args)
}
