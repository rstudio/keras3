k_bincount <-
function (x, weights = NULL, minlength = 0L) 
{
    args <- capture_args2(list(x = as_integer, minlength = as_integer))
    do.call(keras$ops$bincount, args)
}
