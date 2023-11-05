initializer_ones <-
function () 
{
    args <- capture_args2(NULL)
    do.call(keras$initializers$Ones, args)
}
