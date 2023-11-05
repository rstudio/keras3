k_zeros <-
function (shape, dtype = NULL) 
{
    args <- capture_args2(list(shape = normalize_shape))
    do.call(keras$ops$zeros, args)
}
