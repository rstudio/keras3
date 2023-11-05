k_count_nonzero <-
function (x, axis = NULL) 
{
    args <- capture_args2(list(axis = as_axis))
    do.call(keras$ops$count_nonzero, args)
}
