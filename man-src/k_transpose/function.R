k_transpose <-
function (x, axes = NULL) 
{
    args <- capture_args2(list(axes = as_axis))
    do.call(keras$ops$transpose, args)
}
