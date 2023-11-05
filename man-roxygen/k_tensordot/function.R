k_tensordot <-
function (x1, x2, axes = 2L) 
{
    args <- capture_args2(list(axes = as_axis))
    do.call(keras$ops$tensordot, args)
}
