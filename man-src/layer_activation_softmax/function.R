layer_activation_softmax <-
function (object, axis = -1L, ...) 
{
    args <- capture_args2(list(axis = as_integer, input_shape = normalize_shape, 
        batch_size = as_integer, batch_input_shape = normalize_shape), 
        ignore = "object")
    create_layer(keras$layers$Softmax, object, args)
}
