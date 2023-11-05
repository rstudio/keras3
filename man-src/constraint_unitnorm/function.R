constraint_unitnorm <-
function (axis = 0L) 
{
    args <- capture_args2(list(axis = as_axis))
    do.call(keras$constraints$UnitNorm, args)
}
