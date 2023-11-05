k_repeat <-
function (x, repeats, axis = NULL) 
{
    args <- capture_args2(list(axis = as_axis))
    do.call(keras$ops$`repeat`, args)
}
