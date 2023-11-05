k_empty <-
function (shape, dtype = NULL) 
{
    args <- capture_args2(list(shape = normalize_shape))
    do.call(keras$ops$empty, args)
}
