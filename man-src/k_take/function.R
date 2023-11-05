k_take <-
function (x, indices, axis = NULL) 
{
    args <- capture_args2(list(axis = as_axis))
    do.call(keras$ops$take, args)
}
