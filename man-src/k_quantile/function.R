k_quantile <-
function (x, q, axis = NULL, method = "linear", keepdims = FALSE) 
{
    args <- capture_args2(list(axis = as_axis))
    do.call(keras$ops$quantile, args)
}
