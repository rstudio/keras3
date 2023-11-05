k_broadcast_to <-
function (x, shape) 
{
    args <- capture_args2(list(shape = normalize_shape))
    do.call(keras$ops$broadcast_to, args)
}
