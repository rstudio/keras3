k_scatter <-
function (indices, values, shape) 
{
    args <- capture_args2(list(shape = normalize_shape))
    do.call(keras$ops$scatter, args)
}
