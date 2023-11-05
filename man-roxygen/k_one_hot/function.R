k_one_hot <-
function (x, num_classes, axis = -1L, dtype = NULL) 
{
    args <- capture_args2(list(x = as_integer, axis = as_axis))
    do.call(keras$ops$one_hot, args)
}
