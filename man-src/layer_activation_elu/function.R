layer_activation_elu <-
function (object, alpha = 1, ...) 
{
    args <- capture_args2(list(input_shape = normalize_shape, 
        batch_size = as_integer, batch_input_shape = normalize_shape), 
        ignore = "object")
    create_layer(keras$layers$ELU, object, args)
}
