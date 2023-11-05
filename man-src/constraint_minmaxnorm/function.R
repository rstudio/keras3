constraint_minmaxnorm <-
function (min_value = 0, max_value = 1, rate = 1, axis = 0L) 
{
    args <- capture_args2(list(axis = as_axis))
    do.call(keras$constraints$MinMaxNorm, args)
}
