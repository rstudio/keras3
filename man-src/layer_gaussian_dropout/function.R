layer_gaussian_dropout <-
function (object, rate, seed = NULL, ...) 
{
    args <- capture_args2(list(seed = as_integer, input_shape = normalize_shape, 
        batch_size = as_integer, batch_input_shape = normalize_shape), 
        ignore = "object")
    create_layer(keras$layers$GaussianDropout, object, args)
}
