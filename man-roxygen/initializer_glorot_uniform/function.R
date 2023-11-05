initializer_glorot_uniform <-
function (seed = NULL) 
{
    args <- capture_args2(list(seed = as_integer))
    do.call(keras$initializers$GlorotUniform, args)
}
