k_expand_dims <-
function (x, axis) 
{
    args <- capture_args2(list(axis = as_axis))
    do.call(keras$ops$expand_dims, args)
}
