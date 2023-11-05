k_roll <-
function (x, shift, axis = NULL) 
{
    args <- capture_args2(list(axis = as_axis))
    do.call(keras$ops$roll, args)
}
