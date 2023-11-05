layer_repeat_vector <-
function (object, n, ...) 
{
    args <- capture_args2(list(n = as_integer, input_shape = normalize_shape, 
        batch_size = as_integer, batch_input_shape = normalize_shape), 
        ignore = "object")
    create_layer(keras$layers$RepeatVector, object, args)
}
