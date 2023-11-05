initializer_random_normal <-
function (mean = 0, stddev = 0.05, seed = NULL) 
{
    args <- capture_args2(list(seed = as_integer))
    do.call(keras$initializers$RandomNormal, args)
}
