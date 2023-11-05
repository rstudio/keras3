k_slice <-
function (inputs, start_indices, shape) 
{
    args <- capture_args2(list(shape = normalize_shape))
    do.call(keras$ops$slice, args)
}
