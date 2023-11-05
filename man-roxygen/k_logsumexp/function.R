k_logsumexp <-
function (x, axis = NULL, keepdims = FALSE) 
{
    args <- capture_args2(list(axis = as_axis))
    do.call(keras$ops$logsumexp, args)
}
