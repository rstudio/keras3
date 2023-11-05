k_unstack <-
function (x, num = NULL, axis = 0L) 
{
    args <- capture_args2(list(axis = as_axis))
    do.call(keras$ops$unstack, args)
}
