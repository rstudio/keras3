k_cross <-
function (x1, x2, axisa = -1L, axisb = -1L, axisc = -1L, axis = NULL) 
{
    args <- capture_args2(list(axisa = as_integer, axisb = as_integer, 
        axisc = as_integer, axis = as_axis))
    do.call(keras$ops$cross, args)
}
