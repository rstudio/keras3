layer_input <-
function (shape = NULL, batch_size = NULL, dtype = NULL, sparse = NULL, 
    batch_shape = NULL, name = NULL, tensor = NULL) 
{
    args <- capture_args2(list(shape = normalize_shape, batch_size = as_integer, 
        input_shape = normalize_shape, batch_input_shape = normalize_shape))
    do.call(keras$layers$Input, args)
}
