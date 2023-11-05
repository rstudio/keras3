k_prod <-
function (x, axis = NULL, keepdims = FALSE, dtype = NULL) 
{
    args <- capture_args2(list(axis = as_axis))
    do.call(keras$ops$prod, args)
}
