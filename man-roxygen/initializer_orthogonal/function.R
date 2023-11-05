initializer_orthogonal <-
function (gain = 1, seed = NULL) 
{
    args <- capture_args2(list(seed = as_integer))
    do.call(keras$initializers$OrthogonalInitializer, args)
}
