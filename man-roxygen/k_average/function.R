k_average <-
function (x, axis = NULL, weights = NULL) 
{
    args <- capture_args2(list(axis = as_axis))
    do.call(keras$ops$average, args)
}
