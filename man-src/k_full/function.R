k_full <-
function (shape, fill_value, dtype = NULL) 
{
    args <- capture_args2(list(shape = normalize_shape))
    do.call(keras$ops$full, args)
}
