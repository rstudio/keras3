constraint_maxnorm <-
function (max_value = 2L, axis = 0L) 
{
    args <- capture_args2(list(max_value = as_integer, axis = as_axis))
    do.call(keras$constraints$MaxNorm, args)
}
