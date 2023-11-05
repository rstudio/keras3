k_argmax <-
function (x, axis = NULL) 
{
    args <- capture_args2(list(axis = as_axis))
    do.call(keras$ops$argmax, args)
}
