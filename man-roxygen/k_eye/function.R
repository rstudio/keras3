k_eye <-
function (N, M = NULL, k = 0L, dtype = NULL) 
{
    args <- capture_args2(list(k = as_integer))
    do.call(keras$ops$eye, args)
}
