initializer_truncated_normal <-
function (mean = 0, stddev = 0.05, seed = NULL) 
{
    args <- capture_args2(list(seed = as_integer))
    do.call(keras$initializers$TruncatedNormal, args)
}
