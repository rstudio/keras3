k_triu <-
function (x, k = 0L) 
{
    args <- capture_args2(list(k = as_integer))
    do.call(keras$ops$triu, args)
}
