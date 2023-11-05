k_trace <-
function (x, offset = 0L, axis1 = 0L, axis2 = 1L) 
{
    args <- capture_args2(list(offset = as_integer, axis1 = as_integer, 
        axis2 = as_integer))
    do.call(keras$ops$trace, args)
}
