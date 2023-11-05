initializer_random_uniform <-
function (minval = -0.05, maxval = 0.05, seed = NULL) 
{
    args <- capture_args2(list(seed = as_integer))
    do.call(keras$initializers$RandomUniform, args)
}
