k_append <-
function (x1, x2, axis = NULL) 
{
    args <- capture_args2(list(axis = as_axis))
    do.call(keras$ops$append, args)
}
