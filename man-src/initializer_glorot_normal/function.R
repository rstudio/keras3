initializer_glorot_normal <-
function (seed = NULL) 
{
    args <- capture_args2(list(seed = as_integer))
    do.call(keras$initializers$GlorotNormal, args)
}
