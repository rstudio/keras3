initializer_zeros <-
function () 
{
    args <- capture_args2(NULL)
    do.call(keras$initializers$Zeros, args)
}
