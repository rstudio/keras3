k_moments <-
function (x, axes, keepdims = FALSE, synchronized = FALSE) 
{
    args <- capture_args2(list(axes = as_axis))
    do.call(keras$ops$moments, args)
}
