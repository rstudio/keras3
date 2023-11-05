k_linspace <-
function (start, stop, num = 50L, endpoint = TRUE, retstep = FALSE, 
    dtype = NULL, axis = 0L) 
{
    args <- capture_args2(list(num = as_integer, axis = as_axis))
    do.call(keras$ops$linspace, args)
}
