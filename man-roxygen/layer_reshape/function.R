layer_reshape <-
function (object, target_shape, ...) 
{
    args <- capture_args2(list(input_shape = normalize_shape, 
        batch_size = as_integer, batch_input_shape = normalize_shape, 
        target_shape = as_integer), ignore = "object")
    create_layer(keras$layers$Reshape, object, args)
}
