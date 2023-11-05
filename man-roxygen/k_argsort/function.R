k_argsort <-
function (x, axis = -1L) 
{
    args <- capture_args2(list(axis = as_axis))
    do.call(keras$ops$argsort, args)
}
