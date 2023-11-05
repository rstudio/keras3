k_logspace <-
function (start, stop, num = 50L, endpoint = TRUE, base = 10L, 
    dtype = NULL, axis = 0L) 
{
    args <- capture_args2(list(num = as_integer, base = as_integer, 
        axis = as_axis))
    do.call(keras$ops$logspace, args)
}
