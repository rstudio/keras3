k_arange <-
function (start, stop = NULL, step = 1L, dtype = NULL) 
{
    args <- capture_args2(list(start = as_integer, stop = as_integer, 
        step = as_integer))
    do.call(keras$ops$arange, args)
}
