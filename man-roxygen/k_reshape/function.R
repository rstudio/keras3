k_reshape <-
function (x, new_shape) 
{
    args <- capture_args2(list(new_shape = normalize_shape))
    do.call(keras$ops$reshape, args)
}
