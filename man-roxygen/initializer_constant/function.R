initializer_constant <-
function (value = 0) 
{
    args <- capture_args2(NULL)
    do.call(keras$initializers$Constant, args)
}
