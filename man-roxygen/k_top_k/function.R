k_top_k <-
function (x, k, sorted = TRUE) 
{
    args <- capture_args2(list(k = as_integer))
    do.call(keras$ops$top_k, args)
}
