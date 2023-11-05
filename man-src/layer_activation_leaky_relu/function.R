layer_activation_leaky_relu <-
function (object, negative_slope = 0.3, ...) 
{
    args <- capture_args2(list(input_shape = normalize_shape, 
        batch_size = as_integer, batch_input_shape = normalize_shape), 
        ignore = "object")
    create_layer(keras$layers$LeakyReLU, object, args)
}
