layer_activation_relu <-
function (object, max_value = NULL, negative_slope = 0, threshold = 0, 
    ...) 
{
    args <- capture_args2(list(input_shape = normalize_shape, 
        batch_size = as_integer, batch_input_shape = normalize_shape), 
        ignore = "object")
    create_layer(keras$layers$ReLU, object, args)
}
