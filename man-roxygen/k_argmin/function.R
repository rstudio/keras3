k_argmin <-
function (x, axis = NULL) 
{
    args <- capture_args2(list(axis = as_axis))
    do.call(keras$ops$argmin, args)
}
