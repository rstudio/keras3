k_in_top_k <-
function (targets, predictions, k) 
{
    args <- capture_args2(list(k = as_integer))
    do.call(keras$ops$in_top_k, args)
}
