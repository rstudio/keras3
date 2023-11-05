k_concatenate <-
function (xs, axis = 0L) 
{
    args <- capture_args2(list(axis = as_axis))
    do.call(keras$ops$concatenate, args)
}
