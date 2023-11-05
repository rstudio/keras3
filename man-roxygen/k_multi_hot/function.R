k_multi_hot <-
function (inputs, num_tokens, axis = -1L, dtype = NULL) 
{
    args <- capture_args2(list(inputs = as_integer, num_tokens = as_integer, 
        axis = as_axis))
    do.call(keras$ops$multi_hot, args)
}
